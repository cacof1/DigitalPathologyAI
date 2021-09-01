mport torch
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

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

dim = (256,256)
vis_level = 0

class Dataset(BaseDataset):

    
    def __init__(
            self, 
            coords,
            labels,
            wsi_object, 
            channels=3,
    ):
        self.wsi = wsi_object
        self.coords = coords
        self.labels = labels        
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
        label = self.labels[i]

        if self.transforms:   
            image = self.transforms(image)
             
        label = torch.tensor(label, dtype=torch.long, device=device)
        
        return image, label#, image_vis
        
    def __len__(self):
        return len(self.labels)

basepath = sys.argv[1]
filename = sys.argv[2]
log_path = sys.argv[3]

coords_file = h5py.File(basepath + 'patches/{}.h5'.format(filename),'r')
wsi_object = WholeSlideImage(basepath + 'wsi/{}.svs'.format(filename))
coords = coords_file['coords']
labels = coords_file['label']

dataset = Dataset(coords=coords, labels = labels, wsi_object=wsi_object)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

dataset_sizes = {'train':train_size,
                 'val':val_size }

train_dataloader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False, num_workers=0)


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
        x, y = batch
        logits, features = self(x)
        loss = F.cross_entropy(logits, y) 
        acc = accuracy(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #return {'loss' : loss, 'y_pred' : logits, 'y_true' : y}
        return loss
     
    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits, features = self(x)
        loss = F.cross_entropy(logits, y) 
        acc = accuracy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #return {'loss' : loss, 'y_pred' : logits, 'y_true' : y}
        return loss
    
    def testing_step(self, batch, batch_idx):

        x, y = batch
        logits, features = self(x)
        loss = F.cross_entropy(logits, y) 
        acc = accuracy(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #return {'loss' : loss, 'y_pred' : logits, 'y_true' : y}
        return loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
 
model = ImageClassifier()

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=log_path,
    filename='{epoch:02d}-{val_acc:.2f}',
    save_top_k=1,
    mode='max',
)

trainer = pl.Trainer(gpus=1, max_epochs=3,callbacks=[checkpoint_callback])  

trainer.fit(model, train_dataloader, val_dataloader)
