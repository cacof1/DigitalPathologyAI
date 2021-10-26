import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path

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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, coords_file, wsi_file, dim = (256,256), vis_level = 0, inference=False, transform=None, target_transform = None):
        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.coords_file      = coords_file
        self.wsi_file         = wsi_file
        self.vis_level        = vis_level        
        self.dim              = dim
        self.inference        = inference
        self.normalizer       = TorchMacenkoNormalizer()
    def __len__(self):
        return int(self.coords_file.shape[0]/100) ## / 100 for quick processing

    def __getitem__(self, id):
        # load image
        data  = self.coords_file.iloc[id,:]
        image = np.array(self.wsi_file[data["patient_id"]].wsi.read_region([data["coords_x"], data["coords_y"]], self.vis_level, self.dim).convert("RGB"))
        label = data["label"]

        ## Normalization -- not great so far, but buggy otherwise
        try:
            image, H, E  = self.normalizer.normalize(image)
            #image, H, E = MacenkoNormalization(image)
        except:
            pass

        ## Transform - Data Augmentation
        if self.transform: image  = self.transform(image)
        if self.target_transform: label  = self.target_transform(label)

        if(self.inference): return image
        else: return image,label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, coords_file, wsi_file, train_transform = None, val_transform = None, batch_size = 8, **kwargs):
        super().__init__()
        self.batch_size      = batch_size

        ids_split            = np.round(np.array([0.7, 0.8, 1.0])*len(coords_file)).astype(np.int32)
        self.train_data      = DataGenerator(coords_file[:ids_split[0]],              wsi_file,  transform = train_transform, **kwargs)
        self.val_data        = DataGenerator(coords_file[ids_split[0]:ids_split[1]],  wsi_file,  transform = val_transform, **kwargs)
        self.test_data       = DataGenerator(coords_file[ids_split[1]:ids_split[-1]], wsi_file,  transform = val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)

def LoadFileParameter(preprocessingfolder):
    CoordsPath = os.path.join(preprocessingfolder,"patches")
    WSIPath    = os.path.join(preprocessingfolder,"wsi")
    wsi_file = {}
    coords_file = pd.DataFrame()
    source    = Path(CoordsPath).glob("*.csv")
    for filenb,filename in enumerate(source):        
        coords          = pd.read_csv(filename,index_col=0)
        patient_id      = filename.stem
        wsi_file_object = WholeSlideImage(WSIPath + '/{}.svs'.format(patient_id))
        coords['patient_id'] = patient_id
        coords['label']      = 0  
        wsi_file[patient_id] = wsi_file_object

        if(filenb==0): coords_file = coords
        else: coords_file = coords_file.append(coords)

    return wsi_file, coords_file
def SaveFileParameter(df, preprocessingfolder, column_to_add, label_to_add):
    CoordsPath = Path(preprocessingfolder,"patches")
    CoordsPath.mkdir(parents=True, exist_ok=True)
    
    df[label_to_add]  =  pd.Series(column_to_add)    
    df = df.fillna(0)
    for patient_id, df_split in df.groupby(df.patient_id):
        TotalPath = Path(CoordsPath, patient_id+".csv")
        df_split.to_csv(str(TotalPath))

