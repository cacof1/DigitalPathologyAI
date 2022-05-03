import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from wsi_core.WholeSlideImage import WholeSlideImage
import glob
import os
from utils import sampling_schemes

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, coords_file, target="tumour_label", dim_list=[(256, 256)], vis_list=[0],
                 inference=False, transform=None, target_transform=None):

        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.coords = coords_file
        self.vis_list = vis_list
        self.dim_list = dim_list
        self.inference = inference
        self.target = target

    def __len__(self):
        return int(self.coords.shape[0])

    def __getitem__(self, id):
        # load image
        wsi_file  = WholeSlideImage(self.coords["wsi_path"].iloc[id])

        data_dict = {}
        for dim in self.dim_list:
            for vis_level in self.vis_list:
                key = "_".join(map(str,dim))+"_"+str(vis_level)
                data_dict[key]  = np.array(wsi_file.wsi.read_region([self.coords["coords_x"].iloc[id], self.coords["coords_y"].iloc[id]],
                                                                     vis_level, dim).convert("RGB"))

        ## Transform - Data Augmentation

        if self.transform: data_dict = {key: self.transform(value) for (key, value) in data_dict.items()}
        if (self.inference): return data_dict

        else:
            label = int(round(self.coords[self.target].iloc[id]))
            if self.target_transform:
                label = self.target_transform(label)
            return data_dict, label


class DataModule(LightningDataModule):

    def __init__(self, coords_file, train_transform=None, val_transform=None, batch_size=8, n_per_sample=np.Inf,
                 train_size=0.7, val_size=0.3, target=None, sampling_scheme='wsi', **kwargs):
        super().__init__()
        self.batch_size = batch_size

        if sampling_scheme.lower() == 'wsi':
            coords_file_sampled = sampling_schemes.sample_N_per_WSI(coords_file, n_per_sample=n_per_sample)
            svi = np.unique(coords_file_sampled.file_id)
            np.random.shuffle(svi)
            train_idx, val_idx = train_test_split(svi, test_size=val_size, train_size=train_size)
            coords_file_train = coords_file_sampled[coords_file.file_id.isin(train_idx)]
            coords_file_valid = coords_file_sampled[coords_file.file_id.isin(val_idx)]

        elif sampling_scheme.lower() == 'patch':
            coords_file_sampled = sampling_schemes.sample_N_per_WSI(coords_file, n_per_sample=n_per_sample)
            coords_file_train, coords_file_valid = train_test_split(coords_file_sampled,
                                                                    test_size=val_size, train_size=train_size)

        else:  # assume custom split
            sampler = getattr(sampling_schemes, sampling_scheme)
            coords_file_train, coords_file_valid = sampler(coords_file, target=target, n_per_sample=n_per_sample,
                                                           train_size=train_size, test_size=val_size)

        self.train_data = DataGenerator(coords_file_train, transform=train_transform, target=target, **kwargs)
        self.val_data   = DataGenerator(coords_file_valid,   transform=val_transform, target=target, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10, pin_memory=True, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_data,   batch_size=self.batch_size, num_workers=10, pin_memory=True)

def WSIQuery(config, **kwargs):  ## Select based on queries

    dataframe = pd.read_csv(config['DATA']['Mastersheet'])
    for key, item in config['CRITERIA'].items():

        if key == 'id':  # improve robustness if id is sometimes a string, sometimes a float.
            dataframe = dataframe[dataframe[key].astype(str).isin(item)]
        else:
            dataframe = dataframe[dataframe[key].isin(item)]

    ids = dataframe['id'].values
    return ids

def LoadFileParameter(ids, svs_folder, patch_folder):
    coords_file = pd.DataFrame()
    for filenb, file_id in enumerate(ids):
        try:

            PatchPath = Path(patch_folder, '{}.csv'.format(file_id))
            search_WSI_query = os.path.join(svs_folder, '**', str(file_id) + '.svs')
            WSIPath = glob.glob(search_WSI_query, recursive=True)[0]  # if file is hidden recursively

            coords = pd.read_csv(PatchPath, header=0, index_col=0)
            coords = coords.astype({"coords_y": int, "coords_x": int})
            coords['file_id'] = file_id
            coords['wsi_path'] = str(WSIPath)

            if filenb == 0:
                coords_file = coords
            else:
                coords_file = pd.concat([coords_file, coords])
        except:
            print('Unable to find patch data for file {}.csv'.format(file_id))
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