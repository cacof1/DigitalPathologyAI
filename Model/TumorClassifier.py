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
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint

class ImageClassifier(pl.LightningModule):
    
    def __init__(self, num_classes=2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr

        self.backbone = models.resnet50(pretrained=True)
        num_filters = self.backbone.fc.in_features
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        
        self.classifier = torch.nn.Linear(num_filters, self.num_classes)
        
        #self.layer_1 = torch.nn.Linear(num_filters, 1024)
        #self.layer_2 = torch.nn.Linear(1024, 512)
        #self.layer_3 = torch.nn.Linear(512, self.num_classes)
        
    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        #x = self.layer_1(representations)
        #x = F.relu(x)
        #x = self.layer_2(x)
        #x = F.relu(x)
        #x = self.layer_3(x)
        x = self.classifier(representations)
        x = F.softmax(x, dim=1)         
        return [x, representations]

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        image, labels = batch
        print(image.size())
        logits, features = self(image)
        loss = F.cross_entropy(logits, labels) 
        acc = accuracy(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #return {'loss' : loss, 'y_pred' : logits, 'y_true' : y}
        return loss
     
    def validation_step(self, batch, batch_idx):
        
        image, labels = batch
        logits, features = self(image)
        loss = F.cross_entropy(logits, labels) 
        acc = accuracy(logits, labels)        

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #return {'loss' : loss, 'y_pred' : logits, 'y_true' : y}
        return loss
    
    def testing_step(self, batch, batch_idx):

        image, labels = batch
        logits, features = self(image)
        loss = F.cross_entropy(logits, labels) 
        acc = accuracy(logits, labels)        

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #return {'loss' : loss, 'y_pred' : logits, 'y_true' : y}
        return loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        print("hello")
        image, label = batch
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
