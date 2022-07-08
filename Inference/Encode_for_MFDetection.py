import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import toml
import cv2
import sys
import h5py
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.ops import boxes as box_ops
import numpy as np
import torch
import pandas as pd
from Dataloader.DataloaderMitosis import DataGenerator,DataGenerator_Mitosis
import openslide
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, WSIQuery
from torch.utils.data import DataLoader
from torchvision import datasets, models
from Model.AutoEncoder import AutoEncoder
from Model.FasterRCNN import FasterRCNN
import transforms as T
from sklearn.cluster import KMeans
import pytorch_lightning as pl

config_AED = toml.load('Configuration_AED.ini')
Slide_id = [sys.argv[1]]
tile_dataset = LoadFileParameter(Slide_id, config_AED['SVS_Folder'], config_AED['Patches_Folder'])
wsi_object = openslide.open_slide(config_AED['SVS_Folder'] + '/{}.svs'.format(Slide_id[0]))   
dataset = DataGenerator(tile_dataset, transforms=T.Compose([T.ToTensor(),]), inference = True)

Encoder = AutoEncoder.load_from_checkpoint(config_AED['Pretrained_AED'],config = config_AED).encoder
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=32)
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
features_out    = trainer.predict(Encoder,dataloader)
features_out    = np.concatenate(features_out, axis=0)
features_out    = features_out.reshape(features_out.shape[0],-1)
#Clustering
n_clusters = 3
kmeans  = KMeans(n_clusters=n_clusters,verbose=1).fit(features_out)
tile_dataset["labels"] = kmeans.labels_
scores = np.array(tile_dataset.labels.to_list())
scores /= scores.max()
coords = np.array(tile_dataset[["coords_x","coords_y"]])

labels = np.array(tile_dataset.labels.value_counts().index)
def show_patch(coords,label,n):
    fig, axs = plt.subplots(1, n,figsize=(14,2))
    coords = coords[coords['labels']==label]
    coords = coords.sample(n)
    coords.reset_index(inplace=True,drop=True)
    for i in range(n):
        wsi_path = coords['wsi_path'][i]        
        top_left = (coords['coords_x'][i]  ,coords['coords_y'][i]  )
        wsi_object = openslide.open_slide(wsi_path)  
        img = np.array(wsi_object.read_region(top_left, config_AED['vis_level'], config_AED['dim']).convert("RGB"))
        axs[i].imshow(img)
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    
    fig.suptitle('patches with label {}'.format(label))
    plt.show()
    
for label in labels:    
    show_patch(tile_dataset,label =label,n=10)
#Select Targeted Region
target_label = input('Enter the Label of the Targeted Region:/n')
tile_dataset = tile_dataset[tile_dataset.labels == target_label]
tile_dataset.reset_index(inplace=True,drop=True)
tile_dataset.to_csv('{}_with_AED.csv'.format(Slide_id[0]),index=False)
