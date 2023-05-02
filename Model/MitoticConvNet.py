import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision import models
from torch.nn.functional import softmax
import transformers  # from hugging face


# Basic implementation of a convolutional neural network based on common backbones (any in torchvision.models)
class ConvNet(pl.LightningModule):

    def __init__(self, config, label_encoder=None):
        super().__init__()

        self.save_hyperparameters()  # will save the hyperparameters that come as an input.

        self.config = config

        self.backbone = getattr(models, config['BASEMODEL']['Backbone'])
        if 'densenet' in config['BASEMODEL']['Backbone']:
            self.backbone = self.backbone(pretrained=config['ADVANCEDMODEL']['Pretrained'],
                                          drop_rate=config['ADVANCEDMODEL']['Drop_Rate'])
        else:
            self.backbone = self.backbone(pretrained=config['ADVANCEDMODEL']['Pretrained'])

        self.loss_fcn = getattr(torch.nn, self.config["BASEMODEL"]["Loss_Function"])()

        if self.config['BASEMODEL']['Loss_Function'] == 'CrossEntropyLoss':  # there is a bug currently. Quick fix...
            self.loss_fcn = torch.nn.CrossEntropyLoss(label_smoothing=self.config['REGULARIZATION']['Label_Smoothing'])#weight=config['INTERNAL']['weights'])

        self.activation = getattr(torch.nn, self.config["BASEMODEL"]["Activation"])()
        out_feats = list(self.backbone.children())[-1].out_features
        self.model = nn.Sequential(
            self.backbone,
            nn.Linear(out_feats, 512),
            nn.Linear(512, self.config["DATA"]["N_Classes"]),
            self.activation,
        )

        self.LabelEncoder = label_encoder


    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        logits = self.forward(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, labels = val_batch
        logits = self.forward(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def testing_step(self, test_batch, batch_idx):
        image, labels = test_batch
        logits = self.forward(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        image = batch
        output = softmax(self(image), dim=1)

        return self.all_gather(output)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['Algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['Weight_Decay'])
    
        if self.config['SCHEDULER']['Type'] == 'cosine_warmup':
            # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
            # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
            n_steps_per_epoch = self.config['DATA']['N_Training_Examples'] // self.config['BASEMODEL']['Batch_Size']
            total_steps = n_steps_per_epoch * self.config['ADVANCEDMODEL']['Max_Epochs']
            warmup_steps = self.config['SCHEDULER']['Cos_Warmup_Epochs'] * n_steps_per_epoch
            
            sched = transformers.optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=warmup_steps,
                                                                              num_training_steps=total_steps,
                                                                              num_cycles=0.5)  # default lr->0.
            
            scheduler = {'scheduler': sched,
                         'interval': 'step',
                         'frequency': 1}
            
        elif self.config['SCHEDULER']['Type'] == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.config["SCHEDULER"]["Lin_Step_Size"],
                                                        gamma=self.config["SCHEDULER"]["Lin_Gamma"])  
        return ([optimizer], [scheduler])




