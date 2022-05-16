# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:19:55 2021

@author: zhuoy
"""

import h5py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csc_matrix
import scipy.ndimage as scn
from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
import random
import pickle


def get_homography(phh3_image, he_image, num_of_features = 50000):
    
    img1_color = np.array(phh3_image)
    img2_color = np.array(he_image)
    
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
 
    orb_detector = cv2.ORB_create(num_of_features)
 
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = list(matcher.match(d1, d2))
    matches.sort(key = lambda x: x.distance)
    matches = matches[:int(len(matches)*0.99)]
    no_of_matches = len(matches)
 
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
 
    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    #affine = cv2.getAffineTransform(p1, p2)
 
    return homography

def visualize_registration(phh3_image, he_image, homography):
    
    img1_color = np.array(phh3_image)
    img2_color = np.array(he_image)
    
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape 
        
    transformed_img = cv2.warpPerspective(img1_color,homography, (width, height))
    
 
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(transformed_img, alpha, img2_color, beta, 0.0)
    dst = np.uint8(alpha*(transformed_img)+beta*(img2_color))

    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.imshow(img1_color)
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(transformed_img)
    plt.axis('off')
    plt.title('Transformed')
    plt.subplot(2, 2, 3)
    plt.imshow(img2_color)
    plt.axis('off')
    plt.title('Target')
    plt.subplot(2, 2, 4)
    plt.imshow(dst)
    plt.axis('off')
    plt.title('Overlay')
    plt.show()

    return transformed_img

def visualize_coords(coords,wsi_object,vis_level=-1):

        
    if vis_level >1:

        img = wsi_object.wsi.get_thumbnail(wsi_object.level_dim[vis_level])
        img_color = np.array(img)
        plt.imshow(img_color)
        plt.scatter(coords[:,0],coords[:,1],alpha=0.05)      
        #plt.axis('off')
        plt.show()
        
    else:
        print('Figure size out of range')
        
def visualize_mitosis(df, basepath, if_check = True,vis_level=0, dim=(256,256)):
    
    labels = []
    
    for i in range(df.shape[0]):
        filename = df['filename'][i]
        he_top = (df['mitosis_coord_x'][i],df['mitosis_coord_y'][i])
        phh3_top = (df['phh3_coord_x'][i],df['phh3_coord_y'][i])
        index = df['index'][i]

        h_e_object = WholeSlideImage(basepath + '/phh3/h_e/{}.svs'.format(filename))  
        phh3_object = WholeSlideImage(basepath + '/phh3/phh3/{}_pHH3.svs'.format(filename))
        h_e = np.array(h_e_object.wsi.read_region(he_top, vis_level, dim).convert("RGB"))
        phh3 = np.array(phh3_object.wsi.read_region(phh3_top, vis_level, dim).convert("RGB"))
        
        mask = np.load(basepath + '/mitosis_files/{}_mitosis_masks.npy'.format(filename))[index]
        
        masked = np.ma.masked_where(mask == 0, mask)

        plt.subplot(1, 3, 1)
        plt.imshow(h_e)
        plt.imshow(masked,vmin=0,vmax=1, alpha=1)
        plt.title('Slide: {}'.format(filename))
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(h_e)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(phh3)
        plt.axis('off')
        plt.title('No {}/{}'.format(i,df.shape[0]))
    
        plt.show()
        
        if if_check:
            choice = input("Y indicates True, N indicates False:\n")
        
            if choice == 'Y':
                label = 1
            elif choice == 'N':
                label = 0
            else:
                print('Input form incorrect')
            
        
            label = int(label)
        
            labels.append(label)
        
    return labels
        
    
        
