from typing import Dict, Any
from pathlib import Path
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import omero
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset,DataLoader
from pytorch_lightning import LightningDataModule
import openslide
import Utils.sampling_schemes as sampling_schemes
from Utils.OmeroTools import connect, download_image, download_annotation

import matplotlib.pyplot as plt

class DataGenerator(IterableDataset):

    def __init__(self, tile_dataset, config=None,  transform=None, target_transform=None):

        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.tile_dataset     = tile_dataset
        self.vis_list         = config['BASEMODEL']['Vis']
        self.patch_size       = config['BASEMODEL']['Patch_Size']
        self.inference        = config['ADVANCEDMODEL']['Inference']
        self.target           = config['DATA']['Label']

    def __iter__(self):
        for svs_path, svs_group in self.tile_dataset.groupby('SVS_PATH'):
            with openslide.open_slide(svs_path) as slide:
                for idx, row in svs_group.iterrows():
                    x, y, label = row['coords_x'], row['coords_y'], row[self.target]
                    patches = {}
                    for level in self.vis_list:
                        downsample = int(slide.level_downsamples[level])
                        half_patch_size_X = self.patch_size[0]*downsample // 2
                        half_patch_size_Y = self.patch_size[1]*downsample // 2
                        x_start = x - half_patch_size_X
                        y_start = y - half_patch_size_Y
                        patch   = slide.read_region((x_start, y_start), level, self.patch_size).convert('RGB')
                        if self.transform:
                            patch = self.transform(patch)
                            
                        patches[f"{level}"] = patch

                    if self.inference:
                        return patches
                    else:
                        if self.target_transform:
                            label = self.target_transform(label)                        
                        yield patches, label

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
        
        print(tile_dataset_sampled)
        svi = np.unique(tile_dataset_sampled.SVS_PATH)
        np.random.shuffle(svi)

        # Split in train/val/test so that final proportions are train_size/val_size/test_size
        train_val_idx, test_idx = train_test_split(svi, test_size=config['DATA']['Test_Size'])
        train_idx, val_idx      = train_test_split(train_val_idx, test_size=config['DATA']['Val_Size']/(1-config['DATA']['Test_Size']))

        tile_dataset_train = tile_dataset_sampled[tile_dataset_sampled.SVS_PATH.isin(train_idx)]
        tile_dataset_val   = tile_dataset_sampled[tile_dataset_sampled.SVS_PATH.isin(val_idx)]
        tile_dataset_test  = tile_dataset_sampled[tile_dataset_sampled.SVS_PATH.isin(test_idx)]

        self.train_data = DataGenerator(tile_dataset_train, config=config, transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(tile_dataset_val,   config=config, transform=val_transform, **kwargs)
        self.test_data  = DataGenerator(tile_dataset_test,  config=config, transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


def LoadFileParameter(config: dict, SVS_dataset: pd.DataFrame) -> pd.DataFrame:

    cur_basemodel_str = '_'.join(f"{key}_{config['BASEMODEL'][key]}" for key in ['Patch_Size', 'Vis'])
    tile_dataframe = []

    for npy_path, svs_path in zip(SVS_dataset.NPY_PATH,SVS_dataset.SVS_PATH):

        key =  next(iter(np.load(npy_path, allow_pickle=True).item()))
        _, existing_df = np.load(npy_path, allow_pickle=True).item()[key]        ## Temporary because naming in npy arent uniform
        #_, existing_df = np.load(npy_path, allow_pickle=True).item()[cur_basemodel_str]
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

def QueryImageFromCriteria(config: dict, **kwargs) -> pd.DataFrame:
    print("Querying from Server")
    df = pd.DataFrame()
    with connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw']) as conn:
        conn.SERVICE_OPTS.setOmeroGroup('-1')
        keys = list(config['CRITERIA'].keys())
        value_iter = itertools.product(*config['CRITERIA'].values())  ## Create a joint list with all elements
        for value in value_iter:
            query_base = """
            select image.id, image.name, f2.size, a from
            ImageAnnotationLink ial
            join ial.child a
            join ial.parent image
            left outer join image.pixels p
            left outer join image.fileset as fs
            left outer join fs.usedFiles as uf
            left outer join uf.originalFile as f2
            """
            query_end = ""
            params = omero.sys.ParametersI()
            for nb, temp in enumerate(value):
                query_base += "join a.mapValue mv" + str(nb) + " \n        "
                if nb == 0:
                    query_end += "where (mv" + str(nb) + ".name = :key" + str(nb) + " and mv" + str(
                        nb) + ".value = :value" + str(nb) + ")"
                else:
                    query_end += " and (mv" + str(nb) + ".name = :key" + str(nb) + " and mv" + str(
                        nb) + ".value = :value" + str(nb) + ")"
                params.addString('key' + str(nb), keys[nb])
                params.addString('value' + str(nb), temp)

            query = query_base + query_end
            result = conn.getQueryService().projection(query, params, {"omero.group": "-1"})            
            df_criteria = pd.DataFrame()
            if len(result) > 0:
                for row in result:  ## Transform the results into a panda dataframe for each found match
                    temp = pd.DataFrame(
                        [[row[0].val, Path(row[1].val).stem, row[2].val, *row[3].val.getMapValueAsMap().values()]],
                        columns=["id_omero", "id_external", "Size", *row[3].val.getMapValueAsMap().keys()])
                    df_criteria = pd.concat([df_criteria, temp])

                svs_folder = Path(config['DATA']['SVS_Folder'])
                patches_folder = svs_folder / 'patches'
                df_criteria['SVS_PATH'] = [(svs_folder / (image_id + '.svs')).as_posix() for image_id in df_criteria['id_external']]
                df_criteria['NPY_PATH'] = [(patches_folder / (image_id + '.npy')).as_posix() for image_id in df_criteria['id_external']]
                df = pd.concat([df, df_criteria])
    return df

def SynchronizeSVS(config: dict, df: pd.DataFrame) -> None:
    conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])
    conn.SERVICE_OPTS.setOmeroGroup('-1')

    for index, image in df.iterrows():
        filepath = image['SVS_PATH']
        remote_size = image['Size']
        
        if os.path.exists(filepath):  # Exists
            local_size = os.path.getsize(filepath)
            
            if local_size != remote_size:  # Corrupted
                print(f"SVS file size for {filepath} does not match (local: {local_size}, remote: {remote_size}) - redownloading...")
                os.remove(filepath)
                try:
                    download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])
                except Exception as e:
                    print(f"Error downloading image {filepath}: {e}")
        else:  # Doesn't exist
            print(f"SVS file {filepath} does not exist - downloading...")
            try:
                download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])
            except Exception as e:
                print(f"Error downloading image {filepath}: {e}")

    conn.close()

def SynchronizeNPY(config: Dict[str, Any], df: pd.DataFrame) -> None:
    with connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw']) as conn:
        conn.SERVICE_OPTS.setOmeroGroup('-1')

        for index, image in df.iterrows():
            npy_path = image['NPY_PATH']

            if not os.path.exists(npy_path):  # Doesn't exist
                npy_directory = os.path.join(config['DATA']['SVS_Folder'], 'patches')
                os.makedirs(npy_directory, exist_ok=True)

                print(f"NPY file {npy_path} does not exist - downloading...")

                try:
                    download_annotation(conn.getObject("Image", image['id_omero']), npy_directory)
                except Exception as e:
                    print(f"Error downloading NPY file {npy_path}: {e}")
