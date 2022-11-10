import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import Adafactor, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import AdafactorSchedule
    
class ModelWrapper(pl.LightningModule):
    def __init__(self, model, optimizer_type, scheduler_usage, track, loss_type, num_train_steps=False, epochs=False):
        super().__init__()
        self.model = model
        self.optimizer_type = optimizer_type
        self.scheduler_usage = scheduler_usage
        self.track = track
        self.num_train_steps = num_train_steps
        self.epochs = epochs
        self.loss_type = loss_type
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

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['label']
        logits = self.model.forward(x)
        if self.loss_type == 'ce':
            predict_labels = logits.argmax(axis=1)
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

        self.log('train_loss', loss_val)
        if self.scheduler_usage in ['linear', 'cosine', 'adafactor']:
            self.scheduler.step()
        return loss_val
    
    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.train_accuracy.compute(), on_step=False, on_epoch=True)
        self.train_accuracy.reset()
        self.log('train_rocauc_epoch', self.train_rocauc.compute(), on_step=False, on_epoch=True)
        self.train_rocauc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['label']
        logits = self.model.forward(x)
        if self.loss_type == 'ce':
            predict_labels = logits.argmax(axis=1)
        elif self.loss_type == 'bce':
            predict_labels = torch.where(logits > 0.5, 1, 0).flatten()
        elif self.loss_type == 'soft':
            predict_labels = torch.where(logits > 0, 1, 0).flatten()
        self.val_accuracy.update(predict_labels, y)
        val_accuracy = self.val_accuracy(predict_labels, y)

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
        x, y = batch['data'], batch['label']
        logits = self.model.forward(x)
        if self.loss_type == 'ce':
            predict_labels = logits.argmax(axis=1)
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
        
        self.log('test_loss', loss_val)
        self.log('test_rocauc', test_rocauc)
        self.log('test_accuracy', test_accuracy)
        return {
                'loss_val': loss_val, 
                'test_rocauc': test_rocauc, 
                'test_accuracy': test_accuracy
                }

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.val_accuracy.compute(), on_step=False, on_epoch=True)
        self.val_accuracy.reset()
        self.log('valid_rocauc_epoch', self.val_rocauc.compute(), on_step=False, on_epoch=True)
        self.val_rocauc.reset()

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        elif self.optimizer_type == 'adafactor':
            optimizer = Adafactor(self.parameters(), warmup_init=True if self.scheduler_usage == 'adafactor' else False)
        # print(optimizer)
        if self.scheduler_usage == 'reduce' and self.optimizer_type in ['adam', 'sgd']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max' if self.track == 'roc_auc' else 'min')

            return {
                    'optimizer': optimizer, 
                    'lr_scheduler': {
                                     'scheduler': scheduler, 
                                     'monitor': 'valid_rocauc_epoch' if self.track == 'roc_auc' else 'validation_loss'
                                    }
                    }

        elif self.scheduler_usage == 'linear' and self.optimizer_type in ['adam', 'sgd']:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.num_train_steps * self.epochs * 0.1), 
                num_training_steps=self.num_train_steps * self.epochs
            )
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}

        elif self.scheduler_usage == 'cosine' and self.optimizer_type in ['adam', 'sgd']:
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
    