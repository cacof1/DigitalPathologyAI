import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
import torchvision
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import openslide
import h5py, sys, glob
import torch.nn.functional as F


##Losses
def CrossEntropy(output, target):
   log_prob = F.log_softmax(output, dim=1)
   loss = F.nll_loss(log_prob, torch.argmax(target, dim=1), reduction='none')
   return torch.mean(loss)

def SoftDiceLoss(output, target):
   """
   Reference: Milletari, F., Navab, N., & Ahmadi, S. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric
   Medical Image Segmentation. In International Conference on 3D Vision (3DV).                                                                                                                         
   """
   output = F.logsigmoid(output).exp()
   axes = list(range(2, len(output.shape)))
   eps = 1e-10
   intersection = torch.sum(output * target + eps, axes)
   output_sum_square = torch.sum(output * output + eps, axes)
   target_sum_square = torch.sum(target * target + eps, axes)
   sum_squares = output_sum_square + target_sum_square
   return 1.0 - 2.0 * torch.mean(intersection / sum_squares)
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
      mask   = data['mask'].astype(np.uint8)
      mask   = mask[np.newaxis] ## Add one dimension to represent the number of channels
      image  = data['img'].astype(np.float32)
      image  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[np.newaxis]
      #image  = np.moveaxis(image, -1, 0) ## NCHW (Batch size, Channel, Height and Width) ## [C,H,W]
      
      ## Normalization
      
      ## Transform
      if self.transform:
         sample  = self.transform(image=image,mask=mask)
         image, mask = sample['image'], sample['mask']
      return image, mask

### DataLoader
class DataModule(LightningDataModule):
   def __init__(self, filelist, train_transform = None, val_transform = None, batch_size = 8):
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

   def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
   def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
   def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)

## Model
class ModelRegression(LightningModule):
   def __init__(self) -> None:
      super().__init__()
      self.model = smp.FPN("resnet18", in_channels=1, classes=2)#models.segmentation.deeplabv3_resnet50()
      self.loss_fcn = SoftDiceLoss
   
   def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
      return self.model(x)
   
   def training_step(self, batch, batch_idx):
      image, mask = batch
      prediction  = self(image)#['out']
      #print("image",image.min(), image.max())
      #print("mask",mask.min(),mask.max())
      #print("prediction",prediction.min(),prediction.max())
      loss = self.loss_fcn(prediction,mask)
      self.log("loss", loss)
      return loss      

   def validation_step(self, batch, batch_idx):
      image, mask = batch
      prediction = self(image)#['out']
      loss = self.loss_fcn(prediction,mask)
      self.log("val_loss", loss)
      return loss      

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(),lr=1e-1)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
      return [optimizer], [scheduler]
   
## Main
filelist = np.array(glob.glob(sys.argv[1]+"*.npz"))
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = Trainer(gpus=1, max_epochs=5, logger=logger)
model   = ModelRegression()

## Transforms
train_transform = A.Compose([
   A.HorizontalFlip(p=0.5),
   A.Rotate(5),
   A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))

]) ## Note mean of means, mean of stds])
val_transform   = A.Compose([
   A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))
]) ## valdation pipeline just does normalisation and conversion to tensor
data = DataModule(filelist,train_transform = train_transform, val_transform = val_transform, batch_size=16)
trainer.fit(model, data)
images, labels = next(iter(data.train_dataloader()))
grid = torchvision.utils.make_grid(images) 
logger.experiment.add_image('generated_images', grid)
logger.experiment.add_graph(model,images)
logger.experiment.close()
