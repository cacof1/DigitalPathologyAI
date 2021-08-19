
import os
import cv2
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from wsi_core.WholeSlideImage import WholeSlideImage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
from torchvision import datasets, models,transforms
import matplotlib.pyplot as plt

class Dataset(Dataset):
    def __init__(self,coords,wsi_object):
        self.transforms = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
					]
				)

        self.coords = coords
        self.wsi = wsi_object
    def __getitem__(self, i):
        # load images and masks
        vis_level = 0
        dim = (256,256)
        top_left = tuple(coords[i])
        img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))
        
        if self.transforms is not None:
            img= self.transforms(img)
            
        return img

    def __len__(self):
        return self.coords.shape[0]
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

basepath = sys.argv[1]
filename = sys.argv[2]

coords_file = h5py.File(basepath + 'wsi/{}.h5'.format(filename),'r')
wsi_object = WholeSlideImage(basepath + 'wsi/{}.svs'.format(filename))
coords = coords_file['coords']               

vis_level = 0
dim = (256,256)
    
dataset_test = Dataset(coords,wsi_object)

model = torch.load('fasterrcnn_resnet50_fpn')
model.eval()

predictions = []

for pred_id in range(coords.shape[0]):
    x = [dataset_test[pred_id].to(device)]
    predictions = model(x) 
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()
    
    top_left = coords[pred_id]
    img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))

    for i in range(scores.shape[0]):
        box = boxes[i]
        score = scores[i]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        if score > 0.1:
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255*score,255,0),thickness=2)
              
    if pred_id%1000 == 0:
        print('{}/{} images processed'.format(pred_id,coords.shape[0]))
        plt.imshow(img) 
        plt.title('top left coordinate: ({},{})'.format(top_left[0],top_left[1]))
        plt.show() 

    if scores.shape[0] != 0:
        if max(scores) > 0.5:
            print('Mitosis Detected!')
            plt.imshow(img)    
            plt.title('top left coordinate: ({},{})'.format(top_left[0],top_left[1]))
            #plt.savefig('mitosis_in_{}_{}.jpg'.format(filename,pred_id))
            np.savez('mitosis_in_{}_{}.npz'.format(filename,pred_id),top_left = top_left,boxes = boxes,scores=scores)
            plt.show() 
