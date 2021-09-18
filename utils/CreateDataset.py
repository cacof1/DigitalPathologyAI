# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:55:36 2021
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

class AnnotationReader:
    
    '''
    Args:
        wsi_file: Whole slide image with the form of .svs
        coords_file: Coordinations of the patches with the form of .csv
        json_file: Json file including the annotation from pathologists
        annotation_type : Types of annotation to be read, including 'TumourContour','MitoticFigures'
    '''
    
    def __init__(self, coords_file, wsi_file,json_file,annotation_type='TumourContour'):
        
        self.coords_file      = pd.read_csv(coords_file)
        self.wsi_object       = WholeSlideImage(wsi_file)
        
        f = open(json_file,)
        self.annotation       = geojson.load(f)
        f.close()
        
        self.annotation_type  = annotation_type
        
        self.vis_level        = 0        
        self.dim              = (256,256)
        
    def ReadTumourContour(self):
        #Create tumourous labels for all the patches
        if self.annotation_type != 'TumourContour':       
            raise Exception('AnotationTypeError')
            
        else:
            mask = np.zeros(region_size,dtype='int16')
            
            for n in range(len(self.annotation)):

                points = np.array(self.annotation[n]['geometry']['coordinates'])
                points_downsamples = np.int32(points/wsi_object.wsi.level_downsamples[self.vis_level])
                points_downsamples[:,[0,1]] = points_downsamples[:,[1,0]]
                cv2.fillPoly(mask, np.array([points_downsamples], dtype=np.int32), (1))  

            mask = mask.transpose()          
            labels = []
            
            for i in self.coords_file.shape[0]:
                coord_x = self.coords_file['coords_x'][i]
                coord_y = self.coords_file['coords_y'][i]
                
                mask_temp = mask[coord_x:coord_x+self.dim[0],coord_y:coord_y+self.dim[1]]
    
                if mask_temp.mean() < 0.5:
                    label = 0
                else:
                    label = 1
        
                img = np.array(wsi_object.wsi.read_region(tuple(coord_x,coord_y), vis_level, dim).convert("RGB"))
                labels.append(label)
                #masks.append(mask_temp)
    
                if i%1000 == 0:
                    print('{} images processed'.format(i))
    
            labels = np.array(labels)
            #masks = np.array(masks)
            df = self.coords_file
            df['label'] = labels
            
            return df
            #df.to_csv('tumourous{}_{}.csv'.format(self.coords_file['patient_id'][0],self.dim[0]),index=False)
            #np.save('TumourousMasks{}_{}'.format(self.coords_file['patient_id'][0],self.dim[0]), masks)
            
    
    def ReadMitoticFigures(self):   
        #Create bounding boxes for patches with mitotic figure
        if self.annotation_type != 'MitoticFigure':       
            raise Exception('AnotationTypeError')
            
        else:
            x_min1 = []
            x_max1 = []
            y_min1 = []
            y_max1 = []

            x_min2 = []
            x_max2 = []
            y_min2 = []
            y_max2 = []
            
            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            
            for n in range(len(self.annotation)):
                points = np.array(self.annotation[n]['geometry']['coordinates']).squeeze()
                points_downsamples = np.int32(points/wsi_object.wsi.level_downsamples[self.vis_level])

                if len(points) > 10:                          #Circle for Mitotic Figures
                    x_min1.append(int(min(points[:,0])))
                    x_max1.append(int(max(points[:,0])))
                    y_min1.append(int(min(points[:,1])))
                    y_max1.append(int(max(points[:,1])))
       
                else:                                         #Rectangle for Mitotic-like figures
                    x_min2.append(int(min(points[:,0])))
                    x_max2.append(int(max(points[:,0])))
                    y_min2.append(int(min(points[:,1])))
                    y_max2.append(int(max(points[:,1])))
        

            df1['x_min'] = np.array(x_min1)
            df1['x_max'] = np.array(x_max1)
            df1['y_min'] = np.array(y_min1)
            df1['y_max'] = np.array(y_max1)
            df1['label'] = [1] * df1.shape[0]            #Label 1: Mitotic Figures

            df2['x_min'] = np.array(x_min2) 
            df2['x_max'] = np.array(x_max2)
            df2['y_min'] = np.array(y_min2)
            df2['y_max'] = np.array(y_max2)
            df2['label'] = [2] * df2.shape[0]            #Label 2: Mitotic-like figures
            
            
            def get_top_left(df):
    
                coords_m = []
    
                for i in range(df.shape[0]):
                    x = random.randint(df.x_min[i],df.x_max[i])
                    y = random.randint(df.y_min[i],df.y_max[i])
                    top_left = (int(x-(self.dim[0]/2)),int(y-(self.dim[1]/2)))
                    coords_m.append(top_left)  
    
                df['coords_x'] = np.array(coords_m)[:,0]
                df['coords_y'] = np.array(coords_m)[:,1]
    
                return df
            
            df1 = get_top_left(df1)

            if df2.shape[0] == 0:
                df = df1
            else:
                df2 = get_top_left(df2)
                df = pd.concat([df1,df2])
                
            df.drop(df_all[df_all['y_min']==df_all['y_max']].index,inplace=True)
            df.drop(df_all[df_all['x_min']==df_all['x_max']].index,inplace=True)
            df['patient_id'] = [self.coords_file['patient_id'][0]] * df.shape[0]

            df.reset_index(inplace=True,drop=True)
            
            return df
            #df.to_csv('mitosis{}_{}.csv'.format(self.coords_file['patient_id'][0],self.dim[0]),index=False)
            #print('mitosis{}_{}.csv SAVED'.format(self.coords_file['patient_id'][0],self.dim[0]))
            
            
                     
def CreateDataset(filelist, dataset_type = 'TumourClassification'):
    
    '''
    Args:
        filelist: list of the ids of selected patients
        dataset_type: 
            'TumourClassification': return dataset with colunms of ['coords_x','coords_y','patient_id','label']
                      
            'MitosisDetection': return dataset with colunms of ['coords_x','coords_y','x_min','x_max','y_min','y_max','label','patient_id']
             
    '''
    
    if dataset_type == 'TumourClassification':
        
        dfs = []
        
        for filename in filelist:
            coords_file = filename + '.csv'
            wsi_file = filename + '.svs'
            json_file = filename + 'TumourContour.json'
            
            df = AnnotationReader(coords_file,wsi_file,json_file,annotation_type='TumourContour').ReadTumourContour()
            dfs.append(df)
            
        dataset = pd.concat(dfs)
        dataset.reset_index(inplace=True,drop=True)
    
    if dataset_type == 'MitosisDetection':
        
        dfs = []
        
        for filename in filelist:
            coords_file = filename + '.csv'
            wsi_file = filename + '.svs'
            json_file = filename + 'MitoticFigures.json'
            
            df = AnnotationReader(coords_file,wsi_file,json_file,annotation_type='MitoticFigures').ReadMitoticFigures()
            dfs.append(df)
        
        dataset = pd.concat(dfs)
        dataset.reset_index(inplace=True,drop=True) 
        
    return dataset
            
        
            
            

                
        
