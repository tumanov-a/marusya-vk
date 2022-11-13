import warnings
import os, os.path, sys
import pickle
import argparse
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

import torch
import torchmetrics
import pytorch_lightning as pl

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from datasets import *
from model_factory import *
from model_wrapper import *
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='reduce')
    parser.add_argument('--seeds', type=str, default='True')
    parser.add_argument('--track', default='loss')
    parser.add_argument('--sync_bn', default='False')
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='bce')
    parser.add_argument('--add_feat', type=str, default='False')
    parser.add_argument('--add_resp', type=str, default='False')
    parser.add_argument('--rewrite_data', type=str, default='False')

    args = parser.parse_args()
    
    model_type = args.model
    epochs = args.epochs
    batch_size = args.batch
    name = args.name
    device_id = args.device
    optimizer_type = args.optimizer
    scheduler_usage = args.scheduler
    set_seeds = eval(args.seeds)
    track = args.track
    sync_batchnorm = eval(args.sync_bn)
    accumulate_grad_batches = args.accum
    loss_type = args.loss_type
    add_features = eval(args.add_feat)
    add_response_token = eval(args.add_resp)
    rewrite_data = eval(args.rewrite_data)
    print(add_features)

    if set_seeds:
        pl.utilities.seed.seed_everything(seed=42, workers=True)

    if rewrite_data:
    
        csv_data = pd.read_csv('punc_train.tsv', sep='\t', header=0)
        csv_data = csv_data[['punc_phrase', 'punc_context', 'label']]
        csv_data.columns = ['phrase', 'context', 'label']

        if add_features:
            data_user_replics = pd.read_csv('readability_data/prep_user_replics_prep_punc_train.tsv', sep='\t')
            data_phrase = pd.read_csv('readability_data/prep_punc_phrase_prep_punc_train.tsv', sep='\t')
            user_replics_cols = ['TTR', 'Количество абзацев', 'Количество слогов', 'Количество слов']
            phrase_cols = ['Среднее количество слов в предложении', 'Существительных', 'Количество слогов', 'RR']
            readability_parse_user_repl = data_user_replics[user_replics_cols]
            readability_parse_user_repl.columns = [column + ' контекста юзера' for column in readability_parse_user_repl.columns]
            readability_parse_phrase = data_phrase[phrase_cols]
            readability_parse_phrase.columns = [column + ' таргетной фразы' for column in readability_parse_phrase.columns]
            merged_data = pd.merge(csv_data.reset_index(), readability_parse_user_repl.reset_index(), on='index')
            merged_data = pd.merge(merged_data, readability_parse_phrase.reset_index(), on='index')
            feature_cols = list(readability_parse_phrase.columns) + list(readability_parse_user_repl.columns)
            csv_data = merged_data.copy()
            csv_data.drop(['index'], axis=1, inplace=True)

        csv_data.drop_duplicates(['context', 'phrase', 'label'], inplace=True)
        csv_data['context'] = csv_data['context'].apply(lambda x: eval(x))
        csv_data['context'] = csv_data['context'].apply(lambda x: delete_emoji(x))

        if add_features:
            csv_data['toxicity_phrase'] = csv_data['phrase'].apply(lambda x: text2toxicity(x, True))
            csv_data['toxicity_user_context'] = csv_data['context'].apply(lambda x: toxicity_of_context(x))
            csv_data['linguistic_acceptability'] = csv_data['phrase'].apply(lambda x: predict_acceptability(x))
            csv_data['str_for_resp_qual'] = csv_data.apply(return_str_for_resp_qual, axis=1)
            csv_data['resp_qual'] = csv_data['str_for_resp_qual'].apply(lambda x: predict_resp_qual(x))
            csv_data['relevance'] = csv_data['resp_qual'].apply(lambda x: x[0]).astype('float64')
            csv_data['specificity'] = csv_data['resp_qual'].apply(lambda x: x[1]).astype('float64')
            csv_data.drop(['resp_qual', 'str_for_resp_qual'], axis=1, inplace=True)
            feature_cols.extend(['toxicity_phrase', 'toxicity_user_context', 'specificity', 'relevance', 'linguistic_acceptability'])
            csv_data.to_csv('add_features_train.tsv', sep='\t', index=False)
    else:
        csv_data = pd.read_csv('add_features_train.tsv', sep='\t')
        csv_data.drop_duplicates(['context', 'phrase'], inplace=True)
        csv_data['context'] = csv_data['context'].apply(lambda x: eval(x))
        feature_cols = csv_data.columns[3:]

    if model_type == 'bert':
        csv_data['prep_context'] = csv_data['context'].apply(lambda x: preprocess_context(x, 'bert'))
        if add_response_token:
            add_token = '[RESPONSE_TOKEN]'
            csv_data['prep_phrase_context'] = csv_data['prep_context'] + add_token + '– ' + csv_data['phrase'] + ' [SEP]'
        else:
            csv_data['prep_phrase_context'] = csv_data['prep_context'] + '– ' + csv_data['phrase'] + ' [SEP]'
    elif model_type == 'roberta':
        csv_data['prep_context'] = csv_data['context'].apply(lambda x: preprocess_context(x, 'roberta'))
        if add_response_token:
            add_token = '[RESPONSE_TOKEN]'
            csv_data['prep_phrase_context'] = csv_data['prep_context'] + add_token + '– ' + csv_data['phrase'] + '</s>'
        else:
            csv_data['prep_phrase_context'] = csv_data['prep_context'] + '– ' + csv_data['phrase'] + '</s>'
    elif model_type == 't5':
        csv_data['prep_context'] = csv_data['context'].apply(lambda x: preprocess_context(x, 't5'))
        csv_data['prep_phrase_context'] = csv_data['prep_context'] + ' <extra_id_0> ' + '– ' + csv_data['phrase'] + ' <extra_id_0>'

    if not add_features:
        csv_data = csv_data[['prep_phrase_context', 'label']]
    else:
        csv_data = csv_data[['prep_phrase_context'] + list(feature_cols) + ['label']]
    print(csv_data)

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    train_data, test_data = train_test_split(csv_data, test_size=0.15, random_state=42, shuffle=True, stratify=csv_data['label'])
    x = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    splits = folds.split(x, y)

    for n_fold, (train_index, val_index) in enumerate(splits):
        x_train, x_val = x.values[train_index], x.values[val_index]
        y_train, y_val = y.values[train_index], y.values[val_index]
    
        if add_features:
            train_data, train_labels, train_features = x_train[:, 0], y_train, x_train[:, 1:]
            val_data, val_labels, val_features = x_val[:, 0], y_val, x_val[:, 1:]
            test_data, test_labels, test_features = x_test[:, 0], y_test, x_test[:, 1:]

            scaler = StandardScaler()
            scaled_train_features = scaler.fit_transform(train_features)
            scaled_test_features = scaler.transform(test_features)
            scaled_val_features = scaler.transform(val_features)

            with open(f'scaler_{n_fold}.pickle', 'wb') as handle:
                pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            train_data, train_labels = x_train[:, 0], y_train
            val_data, val_labels = x_val[:, 0], y_val
            test_data, test_labels = x_test[:, 0], y_test

        if model_type == 'bert': 
            hugg_name = 'sberbank-ai/ruBert-large'
        elif model_type == 't5':
            hugg_name = 'sberbank-ai/ruT5-large'
        elif model_type == 'roberta':
            hugg_name = 'DeepPavlov/xlm-roberta-large-en-ru'
            # hugg_name = 'xlm-roberta-large'

        tokenizer = AutoTokenizer.from_pretrained(hugg_name)
        if add_response_token:
            special_tokens_dict = {'additional_special_tokens': [add_token]}
            tokenizer.add_special_tokens(special_tokens_dict)

        config = {'MAX_LEN': 256, 'tokenizer': tokenizer, 'model': model_type}

        train_dataset = Dataset(train_data, 
                                train_labels, 
                                config=config,
                                features=scaled_train_features if add_features else None
                                )
        val_dataset = Dataset(val_data, 
                            val_labels, 
                            config=config,
                            features=scaled_val_features if add_features else None
                            )
        test_dataset = Dataset(test_data, 
                            test_labels,
                            config=config,
                            features=scaled_test_features if add_features else None
                            )
        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=4)

    # with open('dataloader.pickle', 'wb') as handle:
        # pickle.dump(train_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=4)
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=4)
        early_stop_callback = EarlyStopping(monitor=track, 
                                            min_delta=0.001, 
                                            # patience=2 if track in ['valid_rocauc_epoch', 'valid_f1_epoch'] else 4, 
                                            patience=2,
                                            mode='max' if track in ['valid_rocauc_epoch', 'valid_f1_epoch'] else 'min')

        wandb_logger = WandbLogger(project="marusya-folds", name=name + f'_fold{n_fold}', log_model='all')
        checkpoint_callback = ModelCheckpoint(monitor=track, 
                                            filename=f"best-checkpoint-fold{n_fold}-val_metric{track:.3f}",
                                            mode='max' if track in ['valid_rocauc_epoch', 'valid_f1_epoch'] else 'min', 
                                            dirpath=f'checkpoints/{name}', 
                                            save_weights_only=True)

        trainer = Trainer(max_epochs=epochs, 
                      logger=wandb_logger, 
                      val_check_interval=1.0, 
                      accelerator='gpu', 
                      devices=device_id, 
                      callbacks=[early_stop_callback, checkpoint_callback],
                      sync_batchnorm=sync_batchnorm,
                      gradient_clip_val=2, 
                      gradient_clip_algorithm="value",
                      accumulate_grad_batches=accumulate_grad_batches)
    
        model = create_model(model_type, loss_type, len(list(feature_cols)))

        if add_response_token:
            model.model.resize_token_embeddings(len(tokenizer))

        if optimizer_type == 'adafactor':
            config_optim = {'lr': 1e-3, 'relative_step': False, 'scale_parameter': False}
        else:
            config_optim = None

        if scheduler_usage not in ['cosine', 'linear']:
            wrap_model = ModelWrapper(model, optimizer_type, scheduler_usage, track, loss_type, config_optim)
        else:
            wrap_model = ModelWrapper(model, optimizer_type, scheduler_usage, track, loss_type, config_optim, num_train_steps=len(train_dataloader), epochs=epochs)
        trainer.fit(wrap_model, train_dataloader, val_dataloader)
        trainer.test(wrap_model, test_dataloader, ckpt_path='best')

        with open(f'trainers/trainer_{name}_nfold{n_fold}.pickle', 'wb') as handle:
            pickle.dump(trainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
