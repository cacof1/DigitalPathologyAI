import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pandas as pd

##utils
from utils.StainNorm import normalizeStaining

## Models
from Model.unet import UNet
from Model.VGG_AutoEncoder import AutoEncoderVGG
import cv2
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
### Dataset
class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, coords_file,wsi_file, transform=None):
        super().__init__()
        self.transform    = transform
        self.coords_file  = coords_file
        self.wsi_file     = wsi_file

        self.vis_level    = 0
        self.dim          = (256,256)
    def __len__(self):
        return int(self.coords_file.shape[0]/100)
    
    def __getitem__(self, id):
        # load image
        coords_x,coords_y,patient_id = self.coords_file.iloc[id,:]
        image = np.array(self.wsi_file[patient_id].wsi.read_region([coords_x, coords_y], self.vis_level, self.dim).convert("RGB"))
        ## Normalization
        image, H, E = normalizeStaining(image)
        ## Transform
        if self.transform: image  = self.transform(image)
        return image

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
        self.train_data    = DataGenerator(self.coords_file[:ids_split[0]], self.wsi_file  ,self.train_transform)
        self.val_data      = DataGenerator(self.coords_file[ids_split[0]:ids_split[1]], self.wsi_file, self.val_transform)
        self.test_data     = DataGenerator(self.coords_file[ids_split[1]:ids_split[-1]], self.wsi_file ,self.val_transform)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)

    
## Model
class AutoEncoder(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = UNet(in_channels=3, n_classes=3)
        self.loss_fcn = torch.nn.L1Loss()#reduction="sum")
        #self.loss_fcn  = torch.nn.MSELoss()#reduction="sum")
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)
   
    def training_step(self, batch, batch_idx):
        image = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction, image)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch
        prediction = self.forward(image)
        loss = self.loss_fcn(prediction,image)
        self.log("val_loss", loss)
        return loss
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return [optimizer], [scheduler]

if __name__ == "__main__":   
    ## Transforms
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=(32,32)),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.ColorJitter(),
        #transforms.RandomRotation(5)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),        
        
    ])

    val_transform   = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),        
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    
    invTrans   = transforms.Compose([
        #transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.5, 1./0.5, 1./0.5 ]),
        #transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ], std = [ 1., 1., 1. ]),
        #transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.229, 1./0.224, 1./0.225 ]),
        #transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),        
        torchvision.transforms.ToPILImage()
    ])

    
    ##First create a master loader
    CoordsFolder = sys.argv[1]
    WSIPath      = "Box01/"
    
    wsi_file = {}
    coords_file = pd.DataFrame()
    for filenb,filename in enumerate(glob.glob(CoordsFolder+"*.h5")):
        coords          = np.array(h5py.File(filename, "r")['coords'])
        patient_id      = filename.split("/")[-1][:-3]
        wsi_file_object      = WholeSlideImage(WSIPath + '{}.svs'.format(patient_id))
        
        coords_file_temp              = pd.DataFrame(coords,columns=['coords_x','coords_y'])
        coords_file_temp['patient_id'] = patient_id
        wsi_file[patient_id] = wsi_file_object
        if(filenb==0): coords_file = coords_file_temp
        else: coords_file = coords_file.append(coords_file_temp)
        
        
        
    ## Main
    seed_everything(42) 
    
    ## Callbacks
    callbacks = [
        ModelCheckpoint(dirpath='./',filename="model.{epoch:02d}-{val_loss:.2f}.h5"),
        EarlyStopping(monitor='val_loss')
    ]
    #logger   = TensorBoardLogger("tb_logs", name="my_model")
    trainer  = Trainer(gpus=1, max_epochs=10,callbacks=callbacks)
    model    = AutoEncoder()
    
    data     = DataModule(coords_file, wsi_file,train_transform = train_transform, val_transform = val_transform, batch_size=8)
    trainer.fit(model, data)
    #torch.save(model,'best_model.pth')
    
    ## Testing
    model              = AutoEncoder.load_from_checkpoint(callbacks[0].best_model_path)
    test_dataset       = DataGenerator(coords_file, wsi_file, transform=val_transform)
    num_of_predictions = 10
    for n in range(num_of_predictions):
        idx        = np.random.randint(len(coords_file),size=1)[0]
        print(idx)
        image     = test_dataset[idx][np.newaxis]
        image_t   = image
        image_t   = np.swapaxes(image_t.squeeze(),-1,0)
        plt.imshow(image_t)
        plt.show()
        image_out = model.forward(image)
        image     = invTrans(image.squeeze())
        image_out = invTrans(image_out.squeeze())
        
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(image_out)
        plt.show()
