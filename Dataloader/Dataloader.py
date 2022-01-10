import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
##Normalization
from Normalization.Macenko import MacenkoNormalization, TorchMacenkoNormalizer

from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import random
import openslide
import sys, glob
import torch.nn.functional as F
from wsi_core.WholeSlideImage import WholeSlideImage

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, coords_file, target="tumour_label", dim=(256, 256), vis_level=0, inference=False, transform=None,
                 target_transform=None):

        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.coords = coords_file
        self.vis_level = vis_level
        self.dim = dim
        self.inference = inference
        self.normalizer = TorchMacenkoNormalizer()
        self.target = target

    def __len__(self):
        return int(self.coords.shape[0])

    def __getitem__(self, id):
        # load image
        wsi_file  = WholeSlideImage(self.coords["wsi_path"].iloc[id])
        data_dict = {}
        data_dict["image"] = np.array(wsi_file.wsi.read_region([self.coords["coords_x"].iloc[id], self.coords["coords_y"].iloc[id]],
                                                               self.vis_level, self.dim).convert("RGB"))
                
        ## Normalization -- not great so far, but buggy otherwise
        # try:
        #    data_dict["image"], H, E  = self.normalizer.normalize(data_dict["image"])
        #    #data_dict["image"], H, E = MacenkoNormalization(data_dict["image"])
        # except:
        #    pass

        ## Transform - Data Augmentation

        if self.transform: data_dict["image"] = self.transform(data_dict["image"])

        if (self.inference):
            return data_dict

        else:  
            label = int(round(self.coords[self.target].iloc[id]))
            if self.target_transform:
                label = self.target_transform(label)
            return data_dict, label


### DataLoader
class DataModule(LightningDataModule):

    def __init__(self, coords_file, train_transform=None, val_transform=None, batch_size=8, n_per_sample=5000,
                 train_size=0.7, val_size=0.25, target=None, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        coords_file = coords_file.groupby("file_id").sample(n=n_per_sample)
        svi = np.unique(coords_file.file_id)
        np.random.shuffle(svi)
        train_idx, val_idx = train_test_split(svi,test_size = val_size, train_size = train_size) #, test_idx = np.split(svi, [int(len(svi)*train_size), 1+int(len(svi)*train_size) + int(len(svi)*val_size)])
        self.train_data = DataGenerator(coords_file[coords_file.file_id.isin(train_idx)], transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(coords_file[coords_file.file_id.isin(val_idx)],   transform=val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10, pin_memory=True, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=10, pin_memory=True)


def WSIQuery(mastersheet, **kwargs):  ## Select based on queries

    dataframe = pd.read_csv(mastersheet)
    for key, item in kwargs.items():
        dataframe = dataframe[dataframe[key] == item]
        
    ids = dataframe['id'].astype('int')
    return sorted(ids)


def LoadFileParameter(ids, svs_folder, patch_folder, fractional_data=1):
    coords_file = pd.DataFrame()
    for filenb, file_id in enumerate(ids):

        try:
            coords = pd.read_csv(patch_folder + '/{}.csv'.format(file_id), index_col=0)
            coords = coords.astype({"coords_y": int, "coords_x": int})

            if fractional_data < 1:
                coords = coords.sample(frac=fractional_data, random_state=42)

            coords['file_id'] = file_id
            coords['wsi_path'] = svs_folder + '/{}.svs'.format(file_id)

            if (filenb == 0):
                coords_file = coords
            else:
                coords_file = coords_file.append(coords)
        except:
            continue

    return coords_file


def SaveFileParameter(df, Patch_Folder, column_to_add, label_to_add):
    CoordsPath = Path(Patch_Folder)
    CoordsPath.mkdir(parents=True, exist_ok=True)
    df[label_to_add] = pd.Series(column_to_add, index=df.index)
    df = df.fillna(0)
    for file_id, df_split in df.groupby(df.file_id):
        TotalPath = Path(CoordsPath, str(file_id) + ".csv")
        df_split.to_csv(str(TotalPath))
