# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:12:27 2021

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
from functions import visualize_mitosis

basepath = sys[1]
filelist = sys[2]

#Check the masks generated from the phh3
for filename in filelist:
    
    df = pd.read_csv(basepath + '/mitosis_files/{}_mitosis_coords.csv'.format(filename))
    df['index'] = df.index
    
    print('Start checking {}'.format(filename))
    df['num_objs'] = visualize_mitosis(df,basepath,if_check=True)
    print('All patches in {} checked'.format(filename))
    df.to_csv(basepath + '/mitosis_files/{}_{}_coords_checked.csv'.format(filename,save_name),index=False)


#Create the dataframe with all mitosis objectives
df_list = []

for filename in filelist:
    df = pd.read_csv(basepath + '/mitosis_files/{}_mitosis_coords_checked.csv'.format(filename))
    df_list.append(df)
    
mitosis_df = pd.concat(df_list)
mitosis_df.reset_index(drop=True,inplace=True)
        
num_objs = visualize_mitosis(mitosis_df,basepath,if_check=False)
mitosis_df.to_csv(basepath + '/mitosis_files/all_mitosis_coords_checked.csv',index=False)
print('Dataframe with {} mitoses created'.format(mitosis_df.shape[0]))
