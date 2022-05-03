# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 08:27:34 2021

@author: zhuoy
"""

import openslide
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import sys

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


class Visualize:
    
    
    def __init__(self, patient_id=None, filename=None):
        
        self.patient_id = patient_id
        self.filename = filename
        self.embeddings = dict()
        
    def create_heatmap(self, vis_level=-1, segment=False):
        
        '''
        Creating the heatmap for visualizing tumour contour
        
        '''
    
        self.df = pd.read_csv(self.filename,index_col=0)
        self.wsi_object = openslide.open_slide(self.patient_id)
        self.preds = (np.array(self.df["tumour_label"]))*100
        self.coords = np.array(self.df[["coords_x","coords_y"]])

        ## broken, heatmap doesnt belong the wsi_object
        #heatmap = self.wsi_object.visHeatmap(self.preds, self.coords, vis_level=vis_level, segment=segment)
        #plt.imshow(heatmap)
        #plt.show()
        
        return heatmap
    
    def plot_components(self,method,n_components):
        
        if n_components==3:
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.embeddings[method][:, 0], self.embeddings[method][:, 1], self.embeddings[method][:, 2])
            ax.set_title('{} Results'.format(method))
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.show()
            
        elif n_components==2:
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
            ax.scatter(self.embeddings[method][:, 0], self.embeddings[method][:, 1])
            ax.set_title('{} Results'.format(method))
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            plt.show()
        
        else:
            raise Exception("n_components is not suitable")
        
    def PCA(self,feature_file,n_components=3, vis_on_app=False):
        '''
        Return and visualize the PCA results of feature vectors
        
        '''
        
        self.features = pd.read_csv(feature_file)
        pca = PCA(n_components=n_components)
        self.embeddings['PCA'] = pca.fit(self.features).transform(self.features)
        
        if n_components==3:
            df_pca = pd.DataFrame(self.embeddings['PCA'],columns=['c1','c2','c3'])
            self.plot_components(method='PCA',n_components=3)
                        
        elif n_components==2:
            df_pca = pd.DataFrame(self.embeddings['PCA'],columns=['c1','c2'])
            self.plot_components(method='PCA',n_components=2)
            
        else:
            raise Exception("n_components is not suitable")
        
        return df_pca
    
    
    def TSNE(self, feature_file,n_components):
        '''
        Return and visualize the TSNE results of feature vectors
        
        '''
        self.features = pd.read_csv(feature_file)
        tsne = TSNE(n_components=n_components, init="pca", learning_rate="auto", random_state=0)
        self.embeddings['TSNE'] = tsne.fit(self.features).transform(self.features)
        
        if n_components==3:
            df_tsne = pd.DataFrame(self.embeddings['TSNE'],columns=['c1','c2','c3'])
            self.plot_components(method='TSNE',n_components=3)
            
            
        elif n_components==2:
            df_tsne = pd.DataFrame(self.embeddings['TSNE'],columns=['c1','c2'])
            self.plot_components(method='TSNE',n_components=2)
            
        else:
            raise Exception("n_components is not suitable")
        
        return df_tsne

    
    def SVD(self, feature_file,n_components):
        '''
        Return and visualize the SVD results of feature vectors
        
        '''
        
        self.features = pd.read_csv(feature_file)
        svd = TruncatedSVD(n_components=n_components)
        self.embeddings['SVD'] = svd.fit(self.features).transform(self.features)
        
        if n_components==3:
            df_svd = pd.DataFrame(self.embeddings['SVD'],columns=['c1','c2','c3'])
            self.plot_components(method='SVD',n_components=3)
            
        elif n_components==2:
            df_svd = pd.DataFrame(self.embeddings['SVD'],columns=['c1','c2'])
            self.plot_components(method='SVD',n_components=2)
            
        else:
            raise Exception("n_components is not suitable")
        
        return df_svd
        
    
    

    
        
        
        
        
        
        
        
        
        
        
        
        
        
