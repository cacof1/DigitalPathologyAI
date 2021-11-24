from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
import pytorch_lightning as pl
import sys
import torch
from torch.optim import Adam
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision import datasets, models, transforms


class ImageClassifier(pl.LightningModule):

    def __init__(self, num_classes=2, lr=0.01, backbone=models.densenet121(), lossfcn=nn.CrossEntropyLoss(), softmax = nn.Identity()):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.backbone = backbone
        self.loss_fcn = lossfcn
        out_feats = list(backbone.children())[-1].out_features
        self.model = nn.Sequential(
            self.backbone,
            nn.Linear(out_feats, 512),
            nn.Linear(512, num_classes),
            softmax,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        acc = accuracy(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, labels = val_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def testing_step(self, test_batch, batch_idx):
        image, labels = test_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        image = batch
        return self(image)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        return optimizer


