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

#filelist = ['210002933',
#            '210003807','210004267','210004688',
#            '210004943','210005012','210005099']
            

#['210005170','210005269','210005274','210005340','210005349','210005351',
filelist = ['210005359','210005378','210000001','210000002',
            '210000003','210000004','210000005','210000006',
            '210000007','210000008','210000009','210000010']

save_name = 'loose_mitosis'

df_list = []

for filename in filelist:
    
    df = pd.read_csv('mitosis_files/{}_{}_coords.csv'.format(filename,save_name))
    df['index'] = df.index
    
    print('Start checking {}'.format(filename))
    df['num_objs'] = visualize_mitosis(df,save_name)
    print('All patches in {} checked'.format(filename))
    df.to_csv('mitosis_files/{}_{}_coords_checked.csv'.format(filename,save_name),index=False)
    df_list.append(df)

#%%       
filelist = ['210002933','210003807','210004267','210004688',
            '210004943','210005012','210005099','210005170',
            '210005269','210005274','210005340','210005349','210005351',
            '210005359','210005378','210000001','210000002',
            '210000003','210000004','210000005','210000006',
            '210000007','210000008','210000009','210000010']
df_list = []

for filename in filelist:
    df = pd.read_csv('mitosis_files/{}_{}_coords_checked.csv'.format(filename,save_name))
    df_list.append(df)
    
mitosis_df = pd.concat(df_list)
#mitosis_df['index'] = mitosis_df.index
mitosis_df.reset_index(drop=True,inplace=True)
        
num_objs = visualize_mitosis(mitosis_df,save_name)

mitosis_df.to_csv('mitosis_files/all_mitosis_coords_checked_0104.csv',index=False)