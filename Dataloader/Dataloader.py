import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os

##utils
from utils.StainNorm import normalizeStaining

from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import openslide
import h5py, sys, glob
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, coords_file, wsi_file, transform=None):
        super().__init__()
        self.transform    = transform
        self.coords_file  = coords_file
        self.vis_level    = 0
        self.wsi_file     = wsi_file
        self.dim          = (256,256)
        
    def __len__(self):
        return int(self.coords_file.shape[0]) ## / 100 for quick processing

    def __getitem__(self, id):
        # load image
        coords_x, coords_y, patient_id, label = self.coords_file.iloc[id,:]
        image = np.array(self.wsi_file[patient_id].wsi.read_region([coords_x, coords_y], self.vis_level, self.dim).convert("RGB"))

        ## Normalization        
        #image, H, E = normalizeStaining(image)

        ## Transform
        if self.transform: image  = self.transform(image)
        
        return image, label


### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, coords_file, wsi_file, train_transform = None, val_transform = None, batch_size = 8):
        super().__init__()
        self.coords_file     = coords_file
        self.wsi_file        = wsi_file
        self.train_transform = train_transform
        self.val_transform   = val_transform
        self.batch_size      = batch_size
        self.train_data      = []
        self.val_data        = []
        self.test_data       = []

    def setup(self, stage):

        ids_split          = np.round(np.array([0.7, 0.8, 1.0])*len(self.coords_file)).astype(np.int32)
        self.train_data    = DataGenerator(self.coords_file[:ids_split[0]], self.wsi_file, self.train_transform)
        self.val_data      = DataGenerator(self.coords_file[ids_split[0]:ids_split[1]], self.wsi_file,  self.val_transform)
        self.test_data     = DataGenerator(self.coords_file[ids_split[1]:ids_split[-1]], self.wsi_file,  self.val_transform)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)



def LoadFileParameter(preprocessingfolder):
    CoordsPath = os.path.join(preprocessingfolder,"patches")
    WSIPath      = os.path.join(preprocessingfolder,"wsi")
    wsi_file = {}
    coords_file = pd.DataFrame()
    for filenb,filename in enumerate(glob.glob(CoordsPath+"/*.h5")):
        coords          = np.array(h5py.File(filename, "r")['coords'])
        patient_id      = filename.split("/")[-1][:-3]
        wsi_file_object = WholeSlideImage(WSIPath + '/{}.svs'.format(patient_id))

        coords_file_temp               = pd.DataFrame(coords,columns=['coords_x','coords_y'])
        coords_file_temp['patient_id'] = patient_id
        coords_file_temp['label']      = 0  ## Temporary
        wsi_file[patient_id] = wsi_file_object
        if(filenb==0): coords_file = coords_file_temp
        else: coords_file = coords_file.append(coords_file_temp)

    return wsi_file, coords_file
