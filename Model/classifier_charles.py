import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
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
from sklearn.model_selection import train_test_split




##Prediction visualization
def visualize(**images):

    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
   
### Dataset
class DataGen(torch.utils.data.Dataset):
   def __init__(self, filelist, transform=None):
      super().__init__()
      self.filelist   = filelist
      self.transform  = transform
   def __len__(self):
      return self.filelist.shape[0]
   
   def __getitem__(self, idx):
      data   = np.load(self.filelist[idx])
      value  = data['value'].astype(np.float32)
      value  = value[np.newaxis]
      image  = data['img'].astype(np.float32)
      #image  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[np.newaxis]
      image  = np.moveaxis(image, -1, 0) ## NCHW (Batch size, Channel, Height and Width) ## [C,H,W]
      
      ## Normalization
      
      ## Transform
      #if self.transform:
      #   sample  = self.transform(image=image)#,mask=mask)
      #   image, mask = sample['image']#, sample['mask']
      return image, value

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
class Classifier(LightningModule):
   def __init__(self) -> None:
       super().__init__()
       self.model = models.resnet18(pretrained=True)
       self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
       self.loss_fcn = torch.nn.BCEWithLogitsLoss()#torch.nn.BCELoss()
   
   def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
       return self.model(x)#, dim =1)
   
   def training_step(self, batch, batch_idx):
      image, value = batch
      prediction  = self.forward(image)
      loss = self.loss_fcn(prediction,value)
      self.log("loss", loss)
      return loss      

   def validation_step(self, batch, batch_idx):
      image, value = batch
      prediction = self.forward(image)
      loss = self.loss_fcn(prediction,value)
      self.log("val_loss", loss)
      return loss      

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(),lr=1e-1)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
      return [optimizer], [scheduler]
   
## Main
filelist = np.array(glob.glob(sys.argv[1]+"*.npz"))
seed_everything(42) 

logger   = TensorBoardLogger("tb_logs", name="my_model")
trainer  = Trainer(gpus=1, max_epochs=5, logger=logger,precision=16)
model    = Classifier()
"""
## Transforms
train_transform = A.Compose([
   A.HorizontalFlip(p=0.5),
   A.Rotate(5),
   A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))

]) ## Note mean of means, mean of stds])
val_transform   = A.Compose([
   A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))
]) ## valdation pipeline just does normalisation and conversion to tensor
"""
train_transform = None
val_transform   = None
data = DataModule(filelist,train_transform = train_transform, val_transform = val_transform, batch_size=32)
trainer.fit(model, data)
torch.save(model,'best_model.pth')

images, labels = next(iter(data.train_dataloader()))
grid = torchvision.utils.make_grid(images) 
logger.experiment.add_image('generated_images', grid)
logger.experiment.add_graph(model,images)
logger.experiment.close()


"""
best_model = torch.load('best_model.pth')
test_dataset     = DataGen(data.test_data.filelist,transform=None )
test_dataset_vis = DataGen(data.test_data.filelist,transform=val_transform )

num_of_predictions = 10
for i in range(num_of_predictions):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = image.to(device).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
              image=image_vis, 
              ground_truth_mask=gt_mask, 
              predicted_mask=pr_mask
              )

"""
