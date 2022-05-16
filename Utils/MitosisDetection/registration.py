# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:02:20 2021

@author: zhuoy
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
import random
import pickle
from functions import get_homography, visualize_registration, visualize_coords

basepath = sys[1]
filename = sys[2]
dim = (256,256)

coords_file = pd.read_csv(basepath + 'phh3/h_e/{}/patches/{}.csv'.format(dim[0],filename))
coords_file.drop('Unnamed: 0',axis=1,inplace=True)
coords_file.drop('contours',axis=1,inplace=True)

he_coords = coords_file.to_numpy()

phh3_object = WholeSlideImage(basepath + 'phh3/phh3/{}_pHH3.svs'.format(filename))
h_e_object = WholeSlideImage(basepath + 'phh3/h_e/{}.svs'.format(filename))

vis_level = 2

points_downsamples = np.int32(he_coords/h_e_object.wsi.level_downsamples[vis_level])
visualize_coords(points_downsamples,h_e_object,vis_level)  

level_downsamples = int(h_e_object.wsi.level_downsamples[vis_level])
region = h_e_object.level_dim[vis_level]

phh3_wsi = phh3_object.wsi
h_e_wsi = h_e_object.wsi

whole_he_low = h_e_wsi.get_thumbnail(h_e_wsi.level_dimensions[vis_level])
whole_phh3_low = phh3_wsi.get_thumbnail(phh3_wsi.level_dimensions[vis_level])

overall_homograph = get_homography(whole_he_low, whole_phh3_low)
transformed_whole_phh3_low = visualize_registration(whole_he_low, whole_phh3_low,overall_homograph)

print('Transformed coords on level {}'.format(vis_level))
print(overall_homograph)


trans_coords = np.float32(points_downsamples).reshape(-1,1,2)
trans_coords = np.squeeze(cv2.perspectiveTransform(trans_coords,overall_homograph))

visualize_coords(points_downsamples,phh3_object,vis_level)  
visualize_coords(trans_coords,phh3_object,vis_level)  

phh3_coords = np.array([x * level_downsamples for x in trans_coords])

vis_level = 0
#upper_limit = (100, 80, 80)
upper_limit = (255, 150, 150)
lower_limit = (0,0,0)
count = 0
std = 30
mitosis_coords = []
new_coords = []
masks = []
    
for i in range(he_coords.shape[0]):
    he_coord = he_coords[i]
    phh3_coord = (int(phh3_coords[i][0]), int(phh3_coords[i][1]))
    phh3 = np.array(phh3_object.wsi.read_region(phh3_coord, vis_level, dim).convert("RGB"))
    #h_e = np.array(h_e_object.wsi.read_region(he_coord, vis_level, dim).convert("RGB"))
    
    mitosis_mask = cv2.inRange(phh3, lower_limit, upper_limit)
    
    if np.mean(mitosis_mask) > 1:           
            
        center = np.unravel_index(np.argmax(mitosis_mask, axis=None), mitosis_mask.shape)
        new_c = (int(center[0]-dim[0]/2),int(center[1]-dim[1]/2))
        new_top = (phh3_coord[0]+new_c[1],phh3_coord[1]+new_c[0])
        
        phh3 = np.array(phh3_object.wsi.read_region(new_top, vis_level, dim).convert("RGB"))
        mitosis_mask = cv2.inRange(phh3, black, brown)
        indices = mitosis_mask.nonzero()
            
        if np.std(indices[0]) < std and np.std(indices[1]) < std:
                
            count += 1
                              
            he_top = (he_coord[0]+new_c[1],he_coord[1]+new_c[0])
            h_e = np.array(h_e_object.wsi.read_region(he_top, vis_level, dim).convert("RGB"))
            
            try:
                homography = get_homography(phh3, h_e, num_of_features = 5000)
                transformed_img = cv2.warpPerspective(phh3,homography, (h_e.shape[0], h_e.shape[1]))               
                transformed_mask = cv2.warpPerspective(mitosis_mask,homography, (h_e.shape[0], h_e.shape[1]))
                transformed_indices = transformed_mask.nonzero()
                if np.std(transformed_indices[0]) < std and np.std(transformed_indices[1]) < std:
                    mask = transformed_mask
                else:
                    mask = mitosis_mask
                    
                
            except:
                mask = mitosis_mask
            
            #mask = mitosis_mask
            masked = np.ma.masked_where(mask == 0, mask)

            plt.subplot(1, 3, 1)
            plt.imshow(phh3)
            plt.axis('off')
            plt.title('phh3:{}'.format(new_top))
            plt.subplot(1, 3, 2)
            plt.imshow(h_e)
            plt.imshow(masked,vmin=0,vmax=1, alpha=1)
            plt.title('Overlay')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(h_e)
            plt.axis('off')
            plt.title('No.{}:{}'.format(count,he_top))
            #plt.savefig('masks/{}_{}_{}'.format(filename,new_top[0],new_top[1]))
            plt.show()
            
            new_coords.append(new_top)
            mitosis_coords.append(he_top)
            masks.append(mask)
    
print('{} mitoses found in {}'.format(count,filename))

mitosis_coords = np.array(mitosis_coords)    
new_coords = np.array(new_coords)    
coords_df = pd.DataFrame()
coords_df['mitosis_coord_x'] = mitosis_coords[:,0]
coords_df['mitosis_coord_y'] = mitosis_coords[:,1]
coords_df['phh3_coord_x'] = new_coords[:,0]
coords_df['phh3_coord_y'] = new_coords[:,1]
coords_df['filename'] = [filename]*coords_df.shape[0]
masks = np.array(masks)      

coords_df.to_csv(basepath + '/mitosis_files/{}_mitosis_coords.csv'.format(filename),index=False)
np.save(basepath + '/mitosis_files/{}_mitosis_masks.npy'.format(filename),masks)






