import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path

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
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, coords_file, dim = (256,256), vis_level = 0, inference=False, transform=None, target_transform = None, target = "tumour_label"):
        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.coords           = coords_file
        self.vis_level        = vis_level        
        self.dim              = dim
        self.inference        = inference
        self.normalizer       = TorchMacenkoNormalizer()
        self.target           = target

    def __len__(self):
        return int(self.coords.shape[0])

    def __getitem__(self, id):
        # load image
        wsi_file = WholeSlideImage(self.coords["wsi_path"].iloc[id])        
        image = np.array(wsi_file.wsi.read_region([ self.coords["coords_x"].iloc[id], self.coords["coords_y"].iloc[id]], self.vis_level, self.dim).convert("RGB"))
        
        ## Normalization -- not great so far, but buggy otherwise
        #try:
        #    image, H, E  = self.normalizer.normalize(image)
        #    #image, H, E = MacenkoNormalization(image)
        #except:
        #    pass
        
        ## Transform - Data Augmentation
        if self.transform:
            image = self.transform(image)

        if self.inference:
            return image

        else: ## Inference
            label = int(round(self.coords[self.target].iloc[id]))
            if self.target_transform:
                label = self.target_transform(label)

            return image, label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, coords_file, wsi_file, train_transform=None, val_transform=None, batch_size=8,
                 pin_memory = False, tiles_splitting=None, svs_splitting=False, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        if tiles_splitting is None:  # to avoid using mutable items in module definition
            self.tiles_splitting = [0.65, 0.95, 0.05]  # 0.05 is not used really, can be removed (easy to infer from first two)
        else:
            self.tiles_splitting = tiles_splitting

        # SVS splitting schemes for training/validating/testing:
        # svs_splitting=False: will use data form the same svs to generate training/validating/testing datasets.
        # svs_splitting=True : will use different svs files to generate training/validating/testing datasets, trying to match roughly the proportions of tiles_splitting. Use if you have enough examples. Useful for multi-label classification.

        if svs_splitting is True:
            # Make sure that svs files are roughly balanced throughout classes.
            train_frac = tiles_splitting[0]
            valid_frac = np.diff(tiles_splitting)[0]
            test_frac = 1.0-train_frac-valid_frac
            class_labels = coords_file[kwargs['target']].unique()

            train_list_per_class = []
            valid_list_per_class = list()
            test_list_per_class = list()
            random.seed(42)
            for class_label in class_labels:
                curlist = coords_file[coords_file['sarcoma_label'] == class_label].file_id.unique()
                random.shuffle(curlist)
                ids_split = np.round(np.array(tiles_splitting) * len(curlist)).astype(np.int32)  # 70 / 20 / 10 splitting for now...
                train_list_per_class.append(curlist[:ids_split[0]])
                valid_list_per_class.append(curlist[ids_split[0]:ids_split[1]])
                test_list_per_class.append(curlist[ids_split[1]:ids_split[-1]])

            train_list_per_class = [item for sublist in train_list_per_class for item in sublist]  # list of lists -> list
            valid_list_per_class = [item for sublist in valid_list_per_class for item in sublist]
            test_list_per_class = [item for sublist in test_list_per_class for item in sublist]

            self.train_data = DataGenerator(coords_file[coords_file.file_id.isin(train_list_per_class)].sample(frac=1, random_state=42), wsi_file,  transform=train_transform, **kwargs)
            self.val_data   = DataGenerator(coords_file[coords_file.file_id.isin(valid_list_per_class)].sample(frac=1, random_state=42), wsi_file,  transform=val_transform, **kwargs)
            self.test_data  = DataGenerator(coords_file[coords_file.file_id.isin(test_list_per_class)].sample(frac=1, random_state=42),  wsi_file,  transform=val_transform, **kwargs)

            # split svs into
            # bla
        else:
            # pool patches from all svs together, then sample randomly among this pool.
            ids_split = np.round(np.array(tiles_splitting)*len(coords_file)).astype(np.int32)
            # randomizing...
            copied_df = coords_file.copy()
            copied_df = copied_df.sample(frac=1, random_state=42)
            self.train_data = DataGenerator(copied_df[:ids_split[0]],              wsi_file,  transform=train_transform, **kwargs)
            self.val_data   = DataGenerator(copied_df[ids_split[0]:ids_split[1]],  wsi_file,  transform=val_transform, **kwargs)
            self.test_data  = DataGenerator(copied_df[ids_split[1]:ids_split[-1]], wsi_file,  transform=val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10, pin_memory=False, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=10, pin_memory=False)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)

def WSIQuery(mastersheet, **kwargs):    ## Select based on queries

    dataframe = pd.read_csv(mastersheet)
    for key, item in kwargs.items():
        dataframe = dataframe[dataframe[key] == item]
    ids = dataframe['id'].astype('int')

    return sorted(ids)

def LoadFileParameter(ids,svs_folder, patch_folder, fractional_data=1):
    coords_file = pd.DataFrame()    
    for filenb,file_id in enumerate(ids):
        try:
            coords             = pd.read_csv(patch_folder + '/{}.csv'.format(file_id),index_col=0)
            coords          = coords.astype({"coords_y":int, "coords_x":int})

            if fractional_data<1:
                coords = coords.sample(frac=fractional_data, random_state=42)

            coords['file_id']  = file_id
            coords['wsi_path'] = svs_folder + '/{}.svs'.format(file_id)

            if(filenb==0): coords_file = coords
            else: coords_file = coords_file.append(coords)
        except: continue
        
    return coords_file

def SaveFileParameter(df, Patch_Folder, column_to_add, label_to_add):

    CoordsPath = Path(Patch_Folder)
    CoordsPath.mkdir(parents=True, exist_ok=True)    
    df[label_to_add]  =  pd.Series(column_to_add, index=df.index)
    df = df.fillna(0)
    for file_id, df_split in df.groupby(df.file_id):
        TotalPath = Path(CoordsPath, str(file_id)+".csv")
        df_split.to_csv(str(TotalPath))
