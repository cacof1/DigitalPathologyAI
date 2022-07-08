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
import sys
import pandas as pd
import seaborn as sns
import openslide
## Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint
import toml
from pathlib import Path

 ## Module - Dataloaders
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, WSIQuery

## Module - Models
from Model.AutoEncoder import AutoEncoder


config   = toml.load(sys.argv[1])

##First create a master loader
MasterSheet    = config['DATA']['Mastersheet']
SVS_Folder     = config['DATA']['SVS_Folder']
Patch_Folder = config['DATA']['Patches_Folder']
Pretrained_Model = sys.argv[2]

seed_everything(config['MODEL']['RANDOM_SEED'])
ids              = WSIQuery(config)
tile_dataset      = LoadFileParameter(ids, SVS_Folder, Patch_Folder)

transform = transforms.Compose([
    transforms.ToTensor(),
])

invTrans   = transforms.Compose([
    torchvision.transforms.ToPILImage()
    ])

## Load the previous  model
trainer = pl.Trainer(gpus=1, max_epochs=config['MODEL']['Max_Epochs'],precision=config['MODEL']['Precision'])
trainer.model = AutoEncoder.load_from_checkpoint(Pretrained_Model, config=config)

## Now predict
test_dataset = DataLoader(DataGenerator(tile_dataset, transform= transform, inference = config['MODEL']['inference']), batch_size=config['MODEL']['Batch_Size'], num_workers=0, shuffle=False)


features_out    = trainer.predict(trainer.model,test_dataset)
features_out    = np.concatenate(features_out, axis=0)
features_out    = features_out.reshape(features_out.shape[0],-1)


#kmeans = MiniBatchKMeans(n_clusters=4,batch_size=32).fit(features_out)
n_clusters = 4
kmeans  = KMeans(n_clusters=n_clusters,verbose=1,n_init= 1).fit(features_out)
#df = pd.DataFrame()
tile_dataset["clusters"] = kmeans.labels_
tile_dataset["clusters"] = 255*tile_dataset["clusters"]/n_clusters
print(tile_dataset)

## Testing
"""
for i in range(10):
    plt.figure(figsize=(20, 2*n_clusters))
    for i in range(n_clusters):
        df = tile_dataset[tile_dataset["labels"]==i]
        nImage = 10    
        imageId = np.random.randint(len(df), size=nImage)
        for j,num in enumerate(imageId):
            index = int(df.iloc[num].name)
            img = test_dataset.dataset.__getitem__(index).squeeze().permute(1,2,0)
            ax = plt.subplot(n_clusters, nImage, i*nImage + j + 1)
            plt.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
    plt.show()


trainer.model =  AutoEncoder.load_from_checkpoint(Pretrained_Model)
image_out    = trainer.predict(trainer.model,test_dataset)
n = 10
tmp = iter(test_dataset)
for j in range(n):
    plt.figure(figsize=(20, 4))
    image = next(tmp)
    for i in range(n):
        img      = invTrans(image[i])
        img_out  = invTrans(image_out[j][i])
        ax = plt.subplot(2, n, i + 1)
        if(i==0):ax.set_title("image_in")
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i + 1 + n)
        if(i==0):ax.set_title("image_out")
        plt.imshow(img_out)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

"""
reducer = umap.UMAP()
embedding = reducer.fit_transform(features_out)
embedding.shape
tile_dataset['embedding-one'] = embedding[:,0]
tile_dataset['embedding-two'] = embedding[:,1]
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="clusters",
    palette=sns.color_palette("hls", n_clusters),
    data=tile_dataset,
    legend="full",
    alpha=0.3
)
plt.show()
print(tile_dataset)

coords = np.array(tile_dataset[["coords_x","coords_y"]])
print(coords, tile_dataset["clusters"])
wsi_file = openslide.open_slide(tile_dataset["wsi_path"].iloc[0])
img = wsi_file.visHeatmap(tile_dataset["clusters"],coords,patch_size=(128, 128),segment=False, cmap='jet')
plt.imshow(img)
plt.colorbar()
plt.show()

##PCA
pca = PCA(n_components=n_clusters)
pca_results = pca.fit_transform(features_out)
tile_dataset['pca-one'] = pca_results[:,0]
tile_dataset['pca-two'] = pca_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="clusters",
    palette=sns.color_palette("hls", n_clusters),
    data=tile_dataset,
    legend="full",
    alpha=0.3
)

plt.show()
##TSNE
pca = PCA(n_components=50)
pca_results = pca.fit_transform(features_out)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,n_jobs=10)
tsne_results = tsne.fit_transform(pca_results)

tile_dataset['tsne-2d-one'] = tsne_results[:,0]
tile_dataset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="clusters",
    palette=sns.color_palette("hls", n_clusters),
    data=tile_dataset,
    legend="full",
    alpha=0.3
)

#plt.plot(tsne_results[:,0], tsne_results[:,1],"ko")
plt.show()
n = 10

