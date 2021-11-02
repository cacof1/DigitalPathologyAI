import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import h5py
import sys
import cv2

from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.nn import functional as F
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule

# This is a LightningModule class used for the classification of patches into tumour subtypes.

class SarcomaClassifier(pl.LightningModule):

    def __init__(self, num_classes=2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr

        self.backbone = models.densenet121(pretrained=False)  # minimum input size is 29x29

        num_filters = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Linear(num_filters, self.num_classes)
        # Fine tuning of models @ https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        # Example for resnet50:
        #self.backbone = models.resnet50(pretrained=True)
        #num_filters = self.backbone.fc.in_features
        #layers = list(self.backbone.children())[:-1]
        #self.feature_extractor = torch.nn.Sequential(*layers)
        #self.classifier = torch.nn.Linear(num_filters, self.num_classes)

    def forward(self, x):

        x = self.backbone(x)
        x = F.softmax(x, dim=1)

        return x

    def training_step(self, train_batch, batch_idx):
        # training_step defines the train loop. It is independent of forward

        image, labels = train_batch
        logits = self(image)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):

        image, labels = val_batch
        logits = self(image)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def testing_step(self, test_batch, batch_idx):

        image, labels = test_batch
        logits = self(image)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):

        image, label = batch

        return self(image)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

######################################################################################################################


if __name__ == "__main__":  # To train

    wsi_file, coords_file = LoadFileParameter('../Preprocessing/')  # (sys.argv[1])

    coords_file = coords_file[coords_file["tumour_label"] == 1]  # Conserve only the patches labeled as tumour

    pl.seed_everything(42)  # Select random seed

    transform = transforms.Compose([
        transforms.ToTensor(),  # this also normalizes to [0,1].
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  # Generate transforms according to resnet/densenet documentation

    data = DataModule(coords_file, wsi_file, train_transform=transform, val_transform=transform, batch_size=4,
                        inference=False, dim=(64, 64), target='label')  # Initialize dataset

    model = SarcomaClassifier()  # Initialize model
    #model = SarcomaClassifier.load_from_checkpoint(sys.argv[2]) # to load from a previous checkpoint

    trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=3, progress_bar_refresh_rate=20)  # Initialize trainer

    test = trainer.fit(model, data)  # Train

