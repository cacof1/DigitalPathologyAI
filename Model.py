import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
import torchvision
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import torch
import openslide
import h5py, sys, glob

### Dataset
class DataGen(torch.utils.data.Dataset):
   def __init__(self, filelist, transform=None):
      super().__init__()
      self.filelist   = filelist
      self.transform  = transform
   def __len__(self):
      return self.filelist.shape[0]

   def __getitem__(self, idx):
      data = np.load(self.filelist[idx])
      mask = data['mask']
      image  = data['img']

      image = np.moveaxis(image, -1, 0) ## NCHW (Batch size, Channel, Height and Width)
      image = image[:3,:,:] ## Remove alpha
      ## Normalization
      
      ## Transform
      if self.transform:
         sample    = self.transform(image=image, mask=mask)
         image, mask = sample['image'], sample['mask']
      return image, mask

### DataLoader
class DataModule(LightningDataModule):
   def __init__(self, filelist, train_transform = None, val_transform = None, batch_size = 32):
     super().__init__()
     self.filelist        = filelist
     self.train_transform = train_transform
     self.val_transform   = val_transform         
     self.batch_size      = batch_size
     self.train_data      = []
   def setup(self, stage):
      ids_split          = np.round(np.array([0.7, 0.8, 1.0])*len(self.filelist)).astype(np.int32)
      self.train_data    = DataGen(self.filelist[:ids_split[0]], self.train_transform)
      self.val_data      = DataGen(self.filelist[ids_split[0]:ids_split[1]], self.val_transform)
      self.test_data     = DataGen(self.filelist[ids_split[1]:ids_split[-1]], self.val_transform)

   def train_dataloader(self):
      return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
   def val_dataloader(self):
      return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
   def test_dataloader(self):
      return DataLoader(self.test_data, batch_size=self.batch_size)

## Model
class ModelRegression(LightningModule):
   def __init__(self) -> None:
      super().__init__()
      self.model = smp.FPN("resnet18", in_channels=3, classes=1, encoder_weights='imagenet')#,activation='sigmoid')
      self.loss_fcn = smp.losses.DiceLoss("binary",from_logits=True)
   
   def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
      return self.model(x)
   
   def training_step(self, batch, batch_idx):
      image, mask = batch
      prediction = self(image)
      loss = self.loss_fcn(mask,prediction)
      self.log("loss", loss)
      return loss      

   def validation_step(self, batch, batch_idx):
      image, mask = batch
      prediction = self(image)
      loss = self.loss_fcn(mask,prediction)
      self.log("val_loss", loss)
      return loss      

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(),lr=1e-1)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
      return [optimizer], [scheduler]
   
## Main
filelist = np.array(glob.glob(sys.argv[1]+"*.npz"))
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = Trainer(gpus=1, max_epochs=25, logger=logger)
model   = ModelRegression()

## Transforms
train_transform = A.Compose([A.HorizontalFlip(p=0.5),A.Rotate(5), A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))]) ## Note mean of means, mean of stds])
val_transform = A.Compose([A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))])
## valdation pipeline just does normalisation and conversion to tensor
data    = DataModule(filelist,train_transform = train_transform, val_transform = val_transform, batch_size=32)
trainer.fit(model, data)
