from Dataloader.Dataloader import *
from torchvision import transforms

import matplotlib.pyplot as plt
import lightning as L
from Utils.PreprocessingTools import Preprocessor
import sys
import pandas as pd
from Model.ConvNet_Preprocessing import ConvNet_Preprocessing
from Model.ConvNet import ConvNet
import cupy as cp
import cv2
import numpy as np
from openslide import OpenSlide

## Fake Config file
config = {}
config['DATA'] = {}
config['BASEMODEL'] = {}
config['ADVANCEDMODEL'] = {}

config['DATA']['SVS_Folder'] = './Data'
config['DATA']['Label'] = None
config['BASEMODEL']['Patch_Size'] = [256,256]
config['BASEMODEL']['Batch_Size'] = 32
config['BASEMODEL']['Vis'] = [0]
config['ADVANCEDMODEL']['Inference'] = True

### First Model


SVS_PATH    = sys.argv[1]
SVS_dataset = pd.DataFrame.from_dict({"SVS_PATH":[SVS_PATH], 'id_external':[SVS_PATH]})
WSI_object = openslide.open_slide(SVS_PATH)

## Find edges and split into patches
xmin = 0
xmax = WSI_object.level_dimensions[config['BASEMODEL']['Vis'][0]][0]
ymin = 0
ymax = WSI_object.level_dimensions[config['BASEMODEL']['Vis'][0]][1]

edges_x  = np.arange(xmin, xmax, config['BASEMODEL']['Patch_Size'][0])
edges_y  = np.arange(ymin, ymax, config['BASEMODEL']['Patch_Size'][1])
EX, EY   = np.meshgrid(edges_x, edges_y)
corners  = np.column_stack((EX.flatten(), EY.flatten()))
tile_dataset = pd.DataFrame({'coords_x': corners[:,0], 'coords_y': corners[:,1]})
tile_dataset['SVS_PATH'] = SVS_PATH
tile_dataset = tile_dataset.head(n=1000)

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].                                                                                                                                                                                                                             
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data =  DataLoader(DataGenerator(tile_dataset, config, transform=val_transform),
                   batch_size=config['BASEMODEL']['Batch_Size'],
                   num_workers=4,
                   pin_memory=False,
                   shuffle=False)


model_preprocessing = ConvNet_Preprocessing.load_from_checkpoint(sys.argv[2])
model_preprocessing.eval()
trainer = L.Trainer(devices=1,
                    accelerator="gpu",
                    benchmark=False, precision='16')    

predictions = trainer.predict(model_preprocessing, data)
predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))
print(predicted_classes_prob.shape)
tissue_names = model_preprocessing.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))
for tissue_no, tissue_name in enumerate(tissue_names):
    tile_dataset['prob_'+ tissue_name] = predicted_classes_prob[:, tissue_no]
    tile_dataset = tile_dataset.fillna(0)

print(tile_dataset)


## Second Model
tile_dataset         = tile_dataset[tile_dataset['prob_Tumour'] > 0.01]
data_classification  =  DataLoader(DataGenerator(tile_dataset, config, transform=val_transform),
                                   batch_size=config['BASEMODEL']['Batch_Size'],
                                   num_workers=4,
                                   pin_memory=False,
                                   shuffle=False)

model_classifier    = ConvNet.load_from_checkpoint(sys.argv[3])
model_classifier.eval()
#compiled_model_classifier = torch.compile(model_classifier)

predictions = trainer.predict(model_classifier, data_classification)
predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))

mesenchymal_tumour_names = model_classifier.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))
print(predicted_classes_prob.shape, mesenchymal_tumour_names)
tumour_dataset = pd.DataFrame()
for tumour_no, tumour_name in enumerate(mesenchymal_tumour_names):
    tumour_dataset['prob_'+tumour_name] = predicted_classes_prob[:, tumour_no]

print(tumour_dataset.mean())
