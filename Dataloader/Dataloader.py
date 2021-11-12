import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from sklearn.utils import shuffle

##Normalization
from Normalization.Macenko import MacenkoNormalization, TorchMacenkoNormalizer

from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import openslide
import sys, glob
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, coords_file, wsi_file, dim = (256,256), vis_level = 0, inference=False, transform=None, target_transform = None, target = "tumour_label"):
        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.coords           = coords_file
        self.wsi_file         = wsi_file
        self.vis_level        = vis_level        
        self.dim              = dim
        self.inference        = inference
        self.normalizer       = TorchMacenkoNormalizer()
        self.target           = target
    def __len__(self):
        return int(self.coords.shape[0]) ## / 100 for quick processing

    def __getitem__(self, id):
        # load image
        image = np.array(self.wsi_file[self.coords["file_id"].iloc[id]].wsi.read_region([ self.coords["coords_x"].iloc[id], self.coords["coords_y"].iloc[id]], self.vis_level, self.dim).convert("RGB"))

        
        ## Normalization -- not great so far, but buggy otherwise
        #try:
        #    image, H, E  = self.normalizer.normalize(image)
        #    #image, H, E = MacenkoNormalization(image)

        #except:
        #    pass

        image = image.astype('float32') / 255.

        ## Transform - Data Augmentation
        if self.transform: image  = self.transform(image)


        if(self.inference):##Training
            return image

        else: ## Inference
            label = self.coords[self.target].iloc[id]
            if self.target_transform: label  = self.target_transform(label)

            return image,label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, coords_file, wsi_file, train_transform=None, val_transform=None, batch_size=8, random_state=0, **kwargs):
        super().__init__()
        self.batch_size       = batch_size

        ids_split             = np.round(np.array([0.7, 0.8, 1.0])*len(coords_file)).astype(np.int32)
        self.train_data       = DataGenerator(coords_file[:ids_split[0]],              wsi_file,  transform = train_transform, **kwargs)
        self.val_data         = DataGenerator(coords_file[ids_split[0]:ids_split[1]],  wsi_file,  transform = val_transform, **kwargs)
        self.test_data        = DataGenerator(coords_file[ids_split[1]:ids_split[-1]], wsi_file,  transform = val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)


def WSIQuery(mastersheet, **kwargs):    ## Select based on queries
    dataframe  = pd.read_csv(mastersheet)
    for key,item in kwargs.items(): dataframe = dataframe[dataframe[key]==item]
    ids = dataframe['id'].astype('int')
    return sorted(ids)


def LoadFileParameter(ids,svs_folder, patch_folder):
    
    wsi_file    = {}
    coords_file = pd.DataFrame()    
    for filenb,file_id in enumerate(ids):
        try:
            coords          = pd.read_csv(patch_folder + '/{}.csv'.format(file_id),index_col=0)
            coords          = coords.astype({"coords_y":int, "coords_x":int})
            wsi_file_object = WholeSlideImage(svs_folder + '/{}.svs'.format(file_id))

            coords['file_id'] = file_id
            wsi_file[file_id] = wsi_file_object
            if(filenb==0): coords_file = coords
            else: coords_file = coords_file.append(coords)

        except: continue

    return wsi_file, coords_file
def SaveFileParameter(df, Patch_Folder, column_to_add, label_to_add):
    CoordsPath = Path(Patch_Folder)
    CoordsPath.mkdir(parents=True, exist_ok=True)
    
    df[label_to_add]  =  pd.Series(column_to_add, index=df.index)
    df = df.fillna(0)
    for file_id, df_split in df.groupby(df.file_id):
        TotalPath = Path(CoordsPath, str(file_id)+".csv")
        df_split.to_csv(str(TotalPath))

