"""
Created on Tue Jul 20 07:25:35 2021

@author: zhuoy
"""

import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csc_matrix
import scipy.ndimage as scn

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson

contours_file = open(sys.argv[1])
coords_file = h5py.File(sys.argv[2], "r+")
wsi_object = WholeSlideImage((sys.argv[3]))
contours = geojson.load(contours_file)
coords = coords_file['coords']

vis_level = 0
region_size = wsi_object.level_dim[vis_level]

mask = np.zeros(region_size,dtype='int16')

for n in range(len(contours)):
    labels_contours = []
    points = np.array(contours[n]['geometry']['coordinates'])
    points_downsamples = np.int32(points/wsi_object.wsi.level_downsamples[vis_level])
    points_downsamples[:,[0,1]] = points_downsamples[:,[1,0]]
    cv2.fillConvexPoly(mask, points_downsamples, (1))  
    '''
    plt.figure(figsize=(10,10))
    plt.imshow(mask)
    plt.show()
    '''
    
labels = []
masks = []
i = 0
dim = (256,256)

for coord in coords:
    
    mask_temp = mask[coord[1]:coord[1]+dim[0],coord[0]:coord[0]+dim[1]]
    
    if mask_temp.mean() < 0.5:
        label = 0
    else:
        label = 1
        
    img = np.array(wsi_object.wsi.read_region((coord[1],coord[0]), vis_level, dim).convert("RGB"))
    labels.append(label)
    masks.append(mask_temp)
    '''
    plt.figure(figsize=(16, 5))
    plt.title('image No.{} label {}'.format(i,label))
    plt.imshow(img)    
    plt.show()    
    plt.imshow(mask_temp)
    plt.show()  
    '''
    i += 1
    
labels = np.array(labels)
masks = np.array(masks)
coords_file.create_dataset("label", data=labels)
coords_file.create_dataset("mask", data=masks)
    
