import warnings
import os, os.path, sys
import pickle
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7,8,9,10'

import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import *
from model_factory import *
from model_wrapper import *

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--resize', type=str, default='False')
    parser.add_argument('--features', type=str)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='reduce')
    parser.add_argument('--seeds', type=str, default='True')
    parser.add_argument('--track', default='roc_auc')
    parser.add_argument('--sync_bn', default='False')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--accum', type=str, default='False')

    args = parser.parse_args()
    
    model_type = args.model
    epochs = args.epochs
    batch_size = args.batch
    name = args.name
    features = args.features
    train_type = args.train_type
    device_id = args.device
    augment = args.augment
    optimizer_type = args.optimizer
    scheduler_usage = args.scheduler
    set_seeds = eval(args.seeds)
    track = args.track
    sync_batchnorm = eval(args.sync_bn)
    loss = args.loss
    accumulate_grad_batches = eval(args.accum)

    if set_seeds:
        pl.utilities.seed.seed_everything(seed=42, workers=True)
    
    all_data, labels = create_data()

    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)
    train_labels, val_labels = train_test_split(labels, test_size=0.2, random_state=42, shuffle=True)

    test_data, val_data = train_test_split(val_data, test_size=0.5, random_state=42, shuffle=True)
    test_labels, val_labels = train_test_split(val_labels, test_size=0.5, random_state=42, shuffle=True)
    
    config = {}

    train_dataset = Dataset(torch.tensor(train_data), 
                                    torch.tensor(train_labels, dtype=torch.long), 
                                    config=config,
                                    model=model_type)
    val_dataset = Dataset(torch.tensor(val_data), 
                                    torch.tensor(val_labels, dtype=torch.long), 
                                    config=config,
                                    model=model_type)
    test_dataset = Dataset(torch.tensor(test_data), 
                                    torch.tensor(test_labels, dtype=torch.long),
                                    config=config,
                                    model=model_type)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=16)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=16)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=16)
    
    early_stop_callback = EarlyStopping(monitor='valid_rocauc_epoch' if track == 'roc_auc' else 'validation_loss', 
                                        min_delta=0.001, 
                                        patience=4 if track == 'roc_auc' else 8, 
                                        mode='max' if track == 'roc_auc' else 'min')

    wandb_logger = WandbLogger(project="marusya-vk", name=name, log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='valid_rocauc_epoch', 
                                          mode='max', 
                                          dirpath=f'checkpoints/{name}', 
                                          save_weights_only=True)

    trainer = Trainer(max_epochs=epochs, 
                      logger=wandb_logger, 
                      val_check_interval=1.0, 
                      accelerator='gpu', 
                      devices=device_id, 
                      callbacks=[early_stop_callback, checkpoint_callback],
                      sync_batchnorm=sync_batchnorm,
                      accumulate_grad_batches=accumulate_grad_batches)
    
    model = create_model(model_type, loss)
    if scheduler_usage not in ['cosine', 'linear']:
        wrap_model = ModelWrapper(model, optimizer_type, scheduler_usage, track, loss)
    else:
        wrap_model = ModelWrapper(model, optimizer_type, scheduler_usage, track, loss, num_train_steps=len(train_dataloader), epochs=epochs)
    trainer.fit(wrap_model, train_dataloader, val_dataloader)
    trainer.test(wrap_model, test_dataloader, ckpt_path='best')
