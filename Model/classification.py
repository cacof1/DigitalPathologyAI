# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:22:35 2021

@author: zhuoy
"""

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

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
import torchvision.transforms as T
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

basepath = 'C:/Users/zhuoy/Note/PathAI/data/'
filename = '484757'
contours_file = open(basepath + 'annotations/{}_annotations.json'.format(filename))
coords_file = h5py.File(basepath + 'wsi/{}.h5'.format(filename),'r')
wsi_object = WholeSlideImage(basepath + 'wsi/{}.svs'.format(filename))
contours = geojson.load(contours_file)
coords = coords_file['coords']
labels = coords_file['label']

dim = (256,256)
vis_level = 0

def visualize(image,label):


    plt.figure(figsize=(16, 5))

    plt.xticks([])
    plt.yticks([])
    plt.title('label {}'.format(label))
    plt.imshow(image)
    plt.show()
    

class Dataset(BaseDataset):

    
    def __init__(
            self, 
            coords,
            wsi_object, 
            channels=3,
            reshape=False,
    ):
        self.wsi = wsi_object
        self.coords = coords
        self.labels = labels        
        self.channels = channels
        self.reshape = reshape
        self.transforms = torch.nn.Sequential(
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #T.Grayscale(num_output_channels=self.channels)

        )
        
    def __getitem__(self, i):
        # read data

        coord = coords[i]
        image = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, dim).convert("RGB"))
        image.astype('float16')
        label = self.labels[i]


        if self.reshape:
            
           image = np.moveaxis(image, -1, 0) ## NCHW (Batch size, Channel, Height and Width)

           image = torch.tensor(image) 
        
           if self.transforms:   

              image = self.transforms(image)
             
        label = torch.tensor(label, dtype=torch.long, device=device)
        #image = image[:1,:,:]
        #print('shape of image:{}'.format(image.shape))
        
        return image, label
        
    def __len__(self):
        return len(self.labels)

dataset = Dataset(coords=coords, wsi_object=wsi_object,reshape=True)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

dataset_sizes = {'train':train_size,
                 'test':test_size }
train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=0)

dataloaders = {'train':train_dataloader,
               'test':test_dataloader }

class_names = ['tumour','normal']

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

epochs = 5
learning_rate = 0.01

optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#%%
if __name__ == '__main__':
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)

#%%
torch.save(model_ft, 'model_{}'.format(filename))
torch.save(model_ft.state_dict(), 'weights_{}'.format(filename))






