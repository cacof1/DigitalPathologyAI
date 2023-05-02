from typing import Dict, Any
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset,DataLoader
from lightning.pytorch import LightningDataModule
import openslide
import Utils.sampling_schemes as sampling_schemes
from monai.data.wsi_reader import WSIReader
import matplotlib.pyplot as plt

##Dataloader  - Monai
class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, tile_dataset, config=None,  transform=None, target_transform=None):

        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.tile_dataset     = tile_dataset
        self.vis_list         = config['BASEMODEL']['Vis']
        self.patch_size       = config['BASEMODEL']['Patch_Size']
        self.inference        = config['ADVANCEDMODEL']['Inference']
        self.target           = config['DATA']['Label']
        self.wsi_reader       = WSIReader(backend=config['DATA']['WSIReader'])
        self.wsi_object_dict: Dict = {}
    def __len__(self):
        return int(self.tile_dataset.shape[0])
    
    def _get_wsi_object(self, image_path):

        if image_path not in self.wsi_object_dict:
            self.wsi_object_dict[image_path] = self.wsi_reader.read(image_path)
        return self.wsi_object_dict[image_path]
    
    def __getitem__(self, id):
        # load image
        svs_path = self.tile_dataset['SVS_PATH'].iloc[id]
        patches = torch.empty((len(self.vis_list), 3, *self.patch_size)) ## [Z, C, W, H]
        wsi_obj = self.wsi_reader.read(svs_path)
        for level in self.vis_list:
            
            downsample = self.wsi_reader.get_downsample_ratio(wsi_obj,level)            
            half_patch_size_X = self.patch_size[0]*downsample // 2
            half_patch_size_Y = self.patch_size[1]*downsample // 2
            x_start = int(self.tile_dataset["coords_x"].iloc[id] - half_patch_size_X)
            y_start = int(self.tile_dataset["coords_y"].iloc[id] - half_patch_size_Y)
            patch, meta   = self.wsi_reader.get_data(wsi=wsi_obj, location=[y_start,x_start], size=self.patch_size, level=level)

            patch = np.swapaxes(patch,0,2)

            if self.transform:
                patch = self.transform(patch)
            patches[level] = patch
        

        if self.inference:
            return patches
        else:
            label = self.tile_dataset[self.target].iloc[id]
            if self.target_transform:
                label = self.target_transform(label)

            return patches, label



class DataModule(LightningDataModule):
    def __init__(self, tile_dataset, train_transform=None, val_transform=None, config = None, label_encoder=None, **kwargs):
        super().__init__()

        self.batch_size  = config['BASEMODEL']['Batch_Size']        
        self.num_workers = int(.8 * mp.Pool()._processes)  # number of workers for dataloader is 80% of maximum workers.

        if label_encoder:
            tile_dataset[config['DATA']['Label']] = label_encoder.transform(tile_dataset[config['DATA']['Label']])  # For classif only
        
        ## Sampling
        if config['DATA']['N_Per_Sample'] is None or config['DATA']['N_Per_Sample'] == float("inf"):
            tile_dataset_sampled = tile_dataset.groupby('SVS_PATH').sample(frac=1)
                
        else:
            tile_dataset_sampled = (
                tile_dataset
                .groupby('SVS_PATH')
                .apply(lambda group: group.sample(min(config['DATA']['N_Per_Sample'], len(group)), replace=False))
                .reset_index(drop=True)
                )

        # Get unique 'SVS_Path' values and split into train val test sets        
        unique_svs_paths                    = tile_dataset_sampled['SVS_PATH'].unique()
        train_val_svs_paths, test_svs_paths = train_test_split(unique_svs_paths,
                                                               train_size= config['DATA']['Train_Size'] + config['DATA']['Val_Size'],
                                                               random_state=42)
        
        train_svs_paths, val_svs_paths      = train_test_split(train_val_svs_paths,
                                                               train_size = config['DATA']['Train_Size']/( config['DATA']['Train_Size'] + config['DATA']['Val_Size']),
                                                               random_state=np.random.randint(0,10000))        
        
        # Create train, val and test datasets based on the 'SVS_Path' values
        tile_dataset_train = tile_dataset_sampled[tile_dataset_sampled['SVS_PATH'].isin(train_svs_paths)]
        tile_dataset_val = tile_dataset_sampled[tile_dataset_sampled['SVS_PATH'].isin(val_svs_paths)]        
        tile_dataset_test = tile_dataset_sampled[tile_dataset_sampled['SVS_PATH'].isin(test_svs_paths)]

        self.train_data = DataGenerator(tile_dataset_train, config=config, transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(tile_dataset_val,   config=config, transform=val_transform, **kwargs)
        self.test_data  = DataGenerator(tile_dataset_test,  config=config, transform=val_transform, **kwargs)
     
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=True)    

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)

def LoadFileParameter(config: dict, SVS_dataset: pd.DataFrame) -> pd.DataFrame:

    cur_basemodel_str = '_'.join(f"{key}_{config['BASEMODEL'][key]}" for key in ['Patch_Size', 'Vis'])
    tile_dataframe = []
    
    for nb, (index, row) in enumerate(SVS_dataset.iterrows()):
        key =  next(iter(np.load(row['NPY_PATH'], allow_pickle=True).item()))
        _, existing_df = np.load(row['NPY_PATH'], allow_pickle=True).item()[key]        ## Temporary because naming in npy arent uniform
        existing_df.sort_index(inplace=True)    
        #_, existing_df = np.load(row['NPY_PATH'], allow_pickle=True).item()[cur_basemodel_str]
        tile_dataframe.append(existing_df)
    tile_dataset = pd.concat(tile_dataframe, axis=0)
    return tile_dataset

def SaveFileParameter(config: dict, df: pd.DataFrame, SVS_ID: str) -> str:
    cur_basemodel_str = '_'.join(f"{key}_{config['BASEMODEL'][key]}" for key in ['Patch_Size', 'Vis'])
    npy_path = Path(config['DATA']['SVS_Folder'], 'patches', f"{SVS_ID}.npy")
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    npy_dict = np.load(npy_path, allow_pickle=True).item() if npy_path.exists() else {}
    npy_dict[cur_basemodel_str] = [config, df]
    np.save(npy_path, npy_dict)
    return str(npy_path)
