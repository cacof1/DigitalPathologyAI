"""
Created on Mon Aug  2 12:33:52 2021

@author: zhuoy
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.functional import softmax
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchmetrics.functional import accuracy
import time
import os
import copy
import h5py
import sys
import cv2

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from wsi_core.WholeSlideImage import WholeSlideImage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

dim = (256,256)
vis_level = 0

class Dataset(BaseDataset):
  
    def __init__(
            self, 
            coords,
            #labels,
            wsi_object, 
            channels=3,
    ):
        self.wsi = wsi_object
        self.coords = coords 
        self.channels = channels
        self.transforms = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
					]
				)
        
    def __getitem__(self, i):
        # read data
        coord = coords[i]
        image_vis = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, dim).convert("RGB"))
        image = image_vis

        if self.transforms:   
            image = self.transforms(image)

        
        return image
        
    def __len__(self):
        return self.coords.shape[0]

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
        
        
    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        x = self.classifier(representations)
        x = F.softmax(x, dim=1)         
        return [x, representations]

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits, features = self(x)
        loss = F.cross_entropy(logits, y) 
        acc = accuracy(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
     
    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits, features = self(x)
        loss = F.cross_entropy(logits, y) 
        acc = accuracy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def testing_step(self, batch, batch_idx):

        x, y = batch
        logits, features = self(x)
        loss = F.cross_entropy(logits, y) 
        acc = accuracy(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

model = ImageClassifier()
log = sys.argv[1]                                  
model = ImageClassifier.load_from_checkpoint(log)  #load the pre-trained model from log
model.freeze()
trainer = pl.Trainer(gpus=1)  

coords_file = h5py.File(sys.argv[2], "r")    #load the h5 file
wsi_object = WholeSlideImage(sys.argv[3])    #load the svs file
coords = coords_file['coords']

dataset = Dataset(coords=coords, wsi_object=wsi_object)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

print('START MAKING PREDICTION ON {}'.format(filename))
preds = trainer.predict(model, dataloader)                              

np.savez('predictions_{}.npz'.format(filename),predictions = preds)       

#load predictions by:
#data = np.load('predictions_{}.npz'.format(filename),allow_pickle=True)
#preds = data['predictions']
#for i in range(preds.shape[0]):
    #preds[i,0] = preds[i,0].cpu().numpy().squeeze()[1]

#predictions = preds[:,0]
#bol = predictions>0.5    #'True' implys there's a tumour patch
