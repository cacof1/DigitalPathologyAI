import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
import numpy as np
import torch
import openslide
import h5py, sys
from wsi_core.WholeSlideImage import WholeSlideImage

### Dataset
class DataGen(torch.utils.data.Dataset):
   def __init__(self, file_path, mask_path, slide_path, transform=None):
      super().__init__()
      self.file_path  = file_path
      self.mask_path  = mask_path
      self.transform  = transform
      self.coords     = h5py.File(file_path, "r")['coords']
      self.wsi_object = WholeSlideImage(slide_path)
   def __len__(self):
      return self.coords.shape[0]

   def __getitem__(self, idx):
     patch = np.array(self.wsi_object.wsi.read_region(tuple(self.coords[idx]), 0, 256))
     mask  = np.zeros(patch.shape)
     ## Normalization
     ## Transform

     if self.transform:
        patch = self.transform(patch)
        mask  = self.transform(mask)
     return image, mask

### DataLoader
class DataModule(LightningDataModule):
  def __init__(self, file_path, mask_path, slide_path,transform = None,batch_size = 32):
    super().__init__()
    self.file_path  = file_path
    self.mask_path  = mask_path
    self.transform  = transform    
    self.batch_size = batch_size
    self.slide_path = slide_path
  def setup(self, stage):
    full_datasets = DataGen(self.file_path, self.mask_path, self.slide_path, self.transform)
    datasplits    = np.round(np.array([0.7, 0.2, 0.1])*len(full_datasets)).astype(np.int32)
    self.train_data, self.val_data,self.test_data =  torch.utils.data.random_split(full_datasets, datasplits)
  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.batch_size)
  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.batch_size)
  def test_dataloader(self):
    return DataLoader(self.test_data, batch_size=self.batch_size)

## Model
class ModelRegression(LightningModule):
    def __init__(self) -> None:
      super().__init__()
      self.model = smp.FPN("resnet18", in_channels=4, classes=2, encoder_weights='imagenet')
      self.loss_fcn = smp.losses.DiceLoss("multiclass", from_logits=True)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
      return self.model(x)
    
    def training_step(self, batch):
      img, mask = batch
      prediction = self.forward(img)
      loss = self.loss_fcn(mask, prediction)
      return loss      

    def validation_step(self, batch):
      img, mask = batch
      prediction = self.forward(img)
      loss = self.loss_fcn(mask, prediction)
      return loss      

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(),lr=1e-1)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
      return [optimizer], [scheduler]


## Main
file_path  = sys.argv[1]
mask_path  = sys.argv[2]
slide_path = sys.argv[3]
trainer = Trainer(gpus=1, max_epochs=5)
model   = ModelRegression()
data    = DataModule(file_path, mask_path,slide_path)
trainer.fit(model, data)
