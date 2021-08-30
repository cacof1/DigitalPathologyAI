import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from unet import UNet
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
class DataGen(torch.utils.data.Dataset):
    def __init__(self, filelist, transform=None):
        super().__init__()
        self.filelist     = filelist
        self.transform    = transform
        self.coords_file  = None
        self.wsi_slide    = None
        
    def __len__(self):
        return self.filelist.shape[0]
    
    def __getitem__(self, idx):
        self.coords_file = h5py.File(sys.argv[2], "r")
        self.wsi_slide   = WholeSlideImage(sys.argv[3])        
        data             = np.load(self.filelist[idx])
        image            = data['img']
        
        ## Normalization
      
        ## Transform
        if self.transform: image  = self.transform(image)
        return image

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, filelist, train_transform = None, val_transform = None, batch_size = 8):
        super().__init__()
        self.filelist        = filelist
        self.train_transform = train_transform
        self.val_transform   = val_transform         
        self.batch_size      = batch_size
        self.train_data      = []
        self.val_data        = []
        self.test_data       = []        
        
    def setup(self, stage):
        
        ids_split          = np.round(np.array([0.7, 0.8, 1.0])*len(self.filelist)).astype(np.int32)
        self.train_data    = DataGen(self.filelist[:ids_split[0]], self.train_transform)
        self.val_data      = DataGen(self.filelist[ids_split[0]:ids_split[1]], self.val_transform)
        self.test_data     = DataGen(self.filelist[ids_split[1]:ids_split[-1]], self.val_transform)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)

    
## Model
class AutoEncoder(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = UNet(in_channels=3, n_classes=3)
        #self.loss_fcn = torch.nn.L1Loss()#reduction="sum")
        self.loss_fcn  = torch.nn.MSELoss()#reduction="sum")
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)
   
    def training_step(self, batch, batch_idx):
        image = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction,image)
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
   
## Transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.RandomResizedCrop(size=(32,32)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #transforms.ColorJitter(),
    #transforms.RandomRotation(5)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform   = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 


invTrans   = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.229, 1./0.224, 1./0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    torchvision.transforms.ToPILImage(),
])
    
## Main
filelist = np.array(glob.glob(sys.argv[1]+"*.npz"))
seed_everything(42) 

## Callbacks
callbacks = [
    ModelCheckpoint(dirpath='./',filename="model.{epoch:02d}-{val_loss:.2f}.h5"),
    EarlyStopping(monitor='val_loss')
    ]

#logger   = TensorBoardLogger("tb_logs", name="my_model")
trainer  = Trainer(gpus=1, max_epochs=5,callbacks=callbacks)
model    = AutoEncoder()

data     = DataModule(filelist,train_transform = train_transform, val_transform = val_transform, batch_size=2)
trainer.fit(model, data)
torch.save(model,'best_model.pth')

## Testing
model   = AutoEncoder.load_from_checkpoint(callbacks[0].best_model_path)
test_dataset     = DataGen(filelist,transform=val_transform)
num_of_predictions = 10
for n in range(num_of_predictions):
    image     = test_dataset[n][np.newaxis] 
    image_out = model.forward(image)
    image     = invTrans(image.squeeze())
    image_out = invTrans(image_out.squeeze())
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(image_out)
    plt.show()
