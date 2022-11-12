import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import Adafactor, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import AdafactorSchedule
from torchmetrics.classification import MultilabelF1Score
    

class ModelWrapper(pl.LightningModule):
    def __init__(self, model, optimizer_type, scheduler_usage, track, loss_type, config_optim=None, num_train_steps=False, epochs=False):
        super().__init__()
        self.model = model
        self.optimizer_type = optimizer_type
        self.scheduler_usage = scheduler_usage
        self.track = track
        self.num_train_steps = num_train_steps
        self.epochs = epochs
        self.loss_type = loss_type
        self.config_optim = config_optim
        if self.loss_type == 'ce':
            self.loss = torch.nn.CrossEntropyLoss()
            self.train_rocauc = torchmetrics.AUROC(num_classes=2)
            self.val_rocauc = torchmetrics.AUROC(num_classes=2)
            self.test_rocauc = torchmetrics.AUROC(num_classes=2)
        elif self.loss_type == 'bce':
            self.loss = torch.nn.BCELoss()
            self.train_rocauc = torchmetrics.AUROC(num_classes=1)
            self.val_rocauc = torchmetrics.AUROC(num_classes=1)
            self.test_rocauc = torchmetrics.AUROC(num_classes=1)
        elif self.loss_type == 'soft':
            self.loss = torch.nn.SoftMarginLoss()
            self.train_rocauc = torchmetrics.AUROC(num_classes=1)
            self.val_rocauc = torchmetrics.AUROC(num_classes=1)
            self.test_rocauc = torchmetrics.AUROC(num_classes=1)
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.train_f1 = torchmetrics.F1Score(average='weighted', num_classes=2)
        self.val_f1 = torchmetrics.F1Score(average='weighted', num_classes=2)
        self.test_f1 = torchmetrics.F1Score(average='weighted', num_classes=2)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, features, y = batch['ids'], batch['att_mask'], batch['features'], batch['target']
        logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, features=features)

        if self.loss_type == 'ce':
            predict_labels = torch.nn.Softmax()(logits).argmax(axis=1)
        elif self.loss_type == 'bce':
            predict_labels = torch.where(logits > 0.5, 1, 0).flatten()
        elif self.loss_type == 'soft':
            predict_labels = torch.where(logits > 0, 1, 0).flatten()

        if self.loss_type == 'ce':
            loss_val = self.loss(logits, y)
            self.train_rocauc.update(logits, y)
            train_rocauc = self.train_rocauc(logits, y)
        elif self.loss_type == 'soft':
            loss_val = self.loss(logits, y.reshape(-1, 1))
            self.train_rocauc.update(logits, y.reshape(-1, 1))
            train_rocauc = self.train_rocauc(logits, y.reshape(-1, 1))
        elif self.loss_type == 'bce':
            loss_val = self.loss(logits, y.reshape(-1, 1).float())
            self.train_rocauc.update(logits, y.reshape(-1, 1))
            train_rocauc = self.train_rocauc(logits, y.reshape(-1, 1))

        self.train_accuracy.update(predict_labels, y)
        train_accuracy = self.train_accuracy(predict_labels, y)
        self.train_f1.update(predict_labels, y)
        train_f1 = self.train_f1(predict_labels, y)

        self.log('train_loss', loss_val)
        if self.scheduler_usage in ['linear', 'cosine', 'adafactor']:
            self.scheduler.step()
        return loss_val
    
    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.train_accuracy.compute(), on_step=False, on_epoch=True)
        self.train_accuracy.reset()
        self.log('train_rocauc_epoch', self.train_rocauc.compute(), on_step=False, on_epoch=True)
        self.train_rocauc.reset()
        self.log('train_f1_epoch', self.train_f1.compute(), on_step=False, on_epoch=True)
        self.train_f1.reset()
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, features, y = batch['ids'], batch['att_mask'], batch['features'], batch['target']
        logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, features=features)
        if self.loss_type == 'ce':
            predict_labels = torch.nn.Softmax()(logits).argmax(axis=1)
        elif self.loss_type == 'bce':
            predict_labels = torch.where(logits > 0.5, 1, 0).flatten()
        elif self.loss_type == 'soft':
            predict_labels = torch.where(logits > 0, 1, 0).flatten()

        self.val_accuracy.update(predict_labels, y)
        val_accuracy = self.val_accuracy(predict_labels, y)
        self.val_f1.update(predict_labels, y)
        val_f1 = self.val_f1(predict_labels, y)

        if self.loss_type == 'ce':
            loss_val = self.loss(logits, y)
            self.val_rocauc.update(logits, y)
            val_rocauc = self.val_rocauc(logits, y)
        elif self.loss_type == 'soft':
            loss_val = self.loss(logits, y.reshape(-1, 1))
            self.val_rocauc.update(logits, y.reshape(-1, 1))
            val_rocauc = self.val_rocauc(logits, y.reshape(-1, 1))
        elif self.loss_type == 'bce':
            loss_val = self.loss(logits, y.reshape(-1, 1).float())
            self.val_rocauc.update(logits, y.reshape(-1, 1))
            val_rocauc = self.val_rocauc(logits, y.reshape(-1, 1))
        
        self.log('validation_loss', loss_val)
        return loss_val
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, features, y = batch['ids'], batch['att_mask'], batch['features'], batch['target']
        logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, features=features)
        if self.loss_type == 'ce':
            predict_labels = torch.nn.Softmax()(logits).argmax(axis=1)
        elif self.loss_type == 'bce':
            predict_labels = torch.where(logits > 0.5, 1, 0).flatten()
        elif self.loss_type == 'soft':
            predict_labels = torch.where(logits > 0, 1, 0).flatten()

        if self.loss_type == 'ce':
            loss_val = self.loss(logits, y)
            test_rocauc = self.test_rocauc(logits, y)
        elif self.loss_type == 'soft':
            loss_val = self.loss(logits, y.reshape(-1, 1))
            test_rocauc = self.test_rocauc(logits, y.reshape(-1, 1))
        elif self.loss_type == 'bce':
            loss_val = self.loss(logits, y.reshape(-1, 1).float())
            test_rocauc = self.test_rocauc(logits, y.reshape(-1, 1))

        test_accuracy = self.test_accuracy(predict_labels, y)
        test_f1 = self.test_f1(predict_labels, y)
        
        self.log('test_loss', loss_val)
        self.log('test_rocauc', test_rocauc)
        self.log('test_accuracy', test_accuracy)
        self.log('test_f1', test_f1)
        return {
                'loss_val': loss_val, 
                'test_rocauc': test_rocauc, 
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                }

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.val_accuracy.compute(), on_step=False, on_epoch=True)
        self.val_accuracy.reset()
        self.log('valid_rocauc_epoch', self.val_rocauc.compute(), on_step=False, on_epoch=True)
        self.val_rocauc.reset()
        self.log('valid_f1_epoch', self.val_f1.compute(), on_step=False, on_epoch=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        elif self.optimizer_type == 'adafactor':
            optimizer = Adafactor(self.parameters(), warmup_init=True if self.scheduler_usage == 'adafactor' else False, lr=self.config_optim['lr'], relative_step=self.config_optim['relative_step'], scale_parameter=self.config_optim['scale_parameter'])
        # print(optimizer)
        if self.scheduler_usage == 'reduce' and self.optimizer_type in ['adam', 'sgd', 'adamw']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max' if self.track in ['valid_rocauc_epoch', 'valid_f1_epoch'] else 'min')

            return {
                    'optimizer': optimizer, 
                    'lr_scheduler': {
                                     'scheduler': scheduler, 
                                     'monitor': self.track
                                    }
                    }

        elif self.scheduler_usage == 'linear' and self.optimizer_type in ['adam', 'sgd', 'adamw']:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.num_train_steps * self.epochs * 0.1), 
                num_training_steps=self.num_train_steps * self.epochs
            )
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}

        elif self.scheduler_usage == 'cosine' and self.optimizer_type in ['adam', 'sgd', 'adamw']:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.num_train_steps * self.epochs * 0.1), 
                num_training_steps=self.num_train_steps * self.epochs
            )
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}

        elif self.scheduler_usage == 'adafactor' and self.optimizer_type == 'adafactor':
            self.scheduler = AdafactorSchedule(optimizer)
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}
            
        return optimizer