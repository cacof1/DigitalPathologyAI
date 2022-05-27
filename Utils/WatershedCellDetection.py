# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:18:59 2022

@author: zhuoy
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import random
import math
from wsi_core.WholeSlideImage import WholeSlideImage
from typing import Union


def plot_images(image_list = [], titles = [], columns = 5, figure_size = (24, 18)):
    count = len(image_list)
    rows = math.ceil(count / columns)

    fig = plt.figure(figsize=figure_size)
    subplots = []
    for index in range(count):
        subplots.append(fig.add_subplot(rows, columns, index + 1))
        if len(titles):
            subplots[-1].set_title(str(titles[index]))
        plt.imshow(image_list[index])
        plt.axis('off')

    plt.show()
    

class WatershedCellDetection:
    
    #Watershed Algorithm for realizing fast cell detection within tiles
    #Input image should be RGB image with color normalization 
    
    def __init__(self, 
                 threshold=0.2, 
                 Ex_iterations=3,
                 dilate_iterations=3,
                 dst_maskSize=5,
                 MaxArea=5000,
                 MinArea=50,
                 MaxR=200,
                 visualize=False):
        
        super(WatershedCellDetection, self).__init__()
        
        self.threshold = threshold
        self.Ex_iterations = Ex_iterations #Number of times erosion and dilation are applied.
        self.dilate_iterations = dilate_iterations #Number of times dilation is applied
        self.dst_maskSize = dst_maskSize #Mask size for distance transform (3 for  DIST_L1 or DIST_C; 3 or 5 for DIST_L2)
        self.MaxArea = MaxArea #The maximum area of a detected cell
        self.MinArea = MinArea #The minimum area of a detected cell
        self.MaxR = MaxR #The maximum value of the R channel (Set for excluding blood cells)
        self.visualize = visualize #Visualize the detection
        
        
    def forward(self, img):

        num_of_cells,selected_cnts = self.count_cells(img)
        
        return num_of_cells
    
    def count_cells(self,img):
        
        image_bgr=cv.cvtColor(img, cv.COLOR_RGB2BGR)
        GrayScaleImage=cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(GrayScaleImage,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations=self.Ex_iterations)
        sure_bg = cv.dilate(opening,kernel,iterations=self.dilate_iterations)

        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,self.dst_maskSize)
        ret, sure_fg = cv.threshold(dist_transform,self.threshold*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv.subtract(sure_bg,sure_fg)

        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0

        image_seg = image_bgr.copy()
        markers = cv.watershed(image_seg,markers)
        image_seg[markers == -1] = [255,0,0]

        contours,hierarchy  = cv.findContours(sure_fg.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        selected_cnts = []

        for cnt in contours:
            if self.select_cnts(img,cnt):
                selected_cnts.append(cnt)
                    
        selected_cnts = tuple(selected_cnts)
        contours_draw = cv.drawContours(img.copy(), selected_cnts, -1, (0,255,0), 1)
        
        if self.visualize:
            plot_images(image_list = [img,image_seg,sure_bg,sure_fg,contours_draw], 
                        titles = ['original','segmentation','sure_bg','sure_fg','contours'], 
                        columns = 2, 
                        figure_size = (8, 12))
            
        num_of_cells = len(selected_cnts)
        print('Number of Detected Cells: {}'.format(num_of_cells))
        
        return num_of_cells,selected_cnts
    
    def select_cnts(self,img,contour):
        
        feature_dict = self.get_cnt_features(img,contour)
        
        if feature_dict['area'] > self.MinArea and feature_dict['area'] < self.MaxArea:
            if feature_dict['mean_val_r'] < self.MaxR:# and feature_dict['mean_val_b'] > 100:
                return True
            
    def get_cnt_features(self,img,contour):
        
        image_bgr=cv.cvtColor(img, cv.COLOR_RGB2BGR)
        GrayScaleImage=cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        
        x,y,w,h = cv.boundingRect(contour)
        aspect_ratio = float(w)/h
        
        area = cv.contourArea(contour)
        
        x,y,w,h = cv.boundingRect(contour)
        rect_area = w*h
        extent = float(area)/rect_area
        
        #hull = cv.convexHull(cnt)
        #hull_area = cv.contourArea(hull)
        #solidity = float(area)/hull_area
        
        #equi_diameter = np.sqrt(4*area/np.pi)
        
        mask = np.zeros(GrayScaleImage.shape,np.uint8)
        cv.drawContours(mask,[contour],0,255,-1)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(GrayScaleImage,mask = mask)
        mean_val, std_val = cv.meanStdDev(image_bgr,mask = mask)
        
        return {'aspect_ratio':aspect_ratio,
                'area':area,
                'extent':extent,
                'mean_val_b':mean_val[0],
                'mean_val_g':mean_val[1],
                'mean_val_r':mean_val[2],
                'std_val_b':std_val[0],
                'std_val_g':std_val[1],
                'std_val_r':std_val[2],
                }
        
    
