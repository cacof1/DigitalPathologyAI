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
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint

class ImageClassifier(pl.LightningModule):
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = models.resnet50(pretrained=True) 
        num_filters = self.backbone.fc.in_features 
        layers = list(self.backbone.children())[:-1] 
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(num_filters, self.num_classes) ## FCN 1024 -> 5
        self.model = nn.Sequential(
            self.feature_extractor,
            self.classifier,
            torch.nn.SoftMax()
            )
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, labels    = batch
        logits, features = self(image)
        loss = self.loss_fcn(logits, labels) 
        acc   = accuracy(logits, labels)        
        return loss
     
    def validation_step(self, batch, batch_idx):
        image, labels = batch
        logits, features = self(image)
        loss = self.loss_fcn(logits, labels) 
        acc = accuracy(logits, labels)        
        return loss
    
    def testing_step(self, batch, batch_idx):
        image, labels = batch
        logits, features = self(image)
        loss = self.loss_fcn(logits, labels) 
        acc = accuracy(logits, labels)        
        return loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        image =  batch
        return self(image)

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
    
if __name__ == "__main__":

    wsi_file, coords_file = LoadFileParameter(sys.argv[1])
    data  = DataModule(coords_file, wsi_file)#, train_transform = train_transform, val_transform = val_transform, batch_size=2)    
    model = ImageClassifier()
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=log_path,
        filename='{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
    mode='max',
    )

    trainer = pl.Trainer(gpus=1, max_epochs=3,callbacks=[checkpoint_callback])      
    trainer.fit(model, data)
