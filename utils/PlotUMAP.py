from Model.AutoEncoder import AutoEncoder, DataGenerator
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from wsi_core.WholeSlideImage import WholeSlideImage
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import sys, glob
import h5py
## Master Loader
CoordsFolder = sys.argv[1]
ModelPath    = sys.argv[2]
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

val_transform   = transforms.Compose([
    #transforms.ToTensor(),
])


invTrans   = transforms.Compose([
])
                                                                                                                                                                                                                                                  
seed_everything(42)
model              = AutoEncoder.load_from_checkpoint(ModelPath)

test_dataset       = DataGenerator(coords_file, wsi_file, transform=val_transform)
num_of_predictions = 10
for n in range(num_of_predictions):
    image     = test_dataset[n][np.newaxis]
    image_t   = image
    image_out = model.forward(image)
    image     = image.squeeze().detach().numpy()
    image_out = image_out.squeeze().detach().numpy()    
    #image_out = invTrans(image_out.squeeze())
    image     = np.swapaxes(image,0,-1)
    image_t   = np.swapaxes(image_t.squeeze(),0,-1)    
    image_out = np.swapaxes(image_out,0,-1)    
    plt.subplot(1,2,1)
    plt.imshow(image_t)
    plt.subplot(1,2,2)
    plt.imshow(image_out)
    plt.show()

