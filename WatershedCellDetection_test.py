# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:01:58 2022

@author: zhuoy
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import ColourNorm
import pandas as pd
import random
import torch
from wsi_core.WholeSlideImage import WholeSlideImage
from WatershedCellDetection import WatershedCellDetection,plot_images

vis_level = 0
dim = (256,256)

wsi_folder = 'C:/Users/zhuoy/Note/PathAI/data/wsi/' 
coords_folder = 'C:/Users/zhuoy/Note/PathAI/data/wsi/patches/'
filename = '485317'
filepath =  os.path.join(wsi_folder, filename + '.svs')
coordspath = os.path.join(coords_folder, filename + '.csv')
wsi_object = WholeSlideImage(filepath)
coords_file = pd.read_csv(coordspath)
coords_file.drop('Unnamed: 0',axis=1,inplace=True)
coords_file.drop('contours',axis=1,inplace=True)
he_coords = coords_file.to_numpy()

N = random.randint(0, he_coords.shape[0])
image = np.array(wsi_object.wsi.read_region(he_coords[N], vis_level, dim).convert("RGB"))
image_tensor = torch.from_numpy(image).permute(2, 0, 1)
TrainedColorNormFile = 'C:/Users/zhuoy/Note/PathAI/MitosisDetection/phh3/StainNorm/210003477_vis0_HERef.pt'
MacenkoNormaliser = ColourNorm.Macenko(saved_fit_file=TrainedColorNormFile)
image_norm, H, E = MacenkoNormaliser.normalize(image_tensor, stains=True)
image_norm = image_norm.cpu().detach().numpy()
image_norm = image_norm.astype('uint8')
image_bgr=cv.cvtColor(image_norm, cv.COLOR_RGB2BGR)
image_gray=cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
#image_gray = np.asarray(image_gray)

plot_images(image_list = [image, image_norm,image_gray], 
            titles = ['Image {}'.format(N),'ColorNorm', 'Gray Scale'], 
            columns = 3, 
            figure_size = (10, 10))

#%%
detector = WatershedCellDetection(threshold=0.2,
                                  visualize=True)

num_of_cells = detector.forward(image)