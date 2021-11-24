import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csc_matrix
import scipy.ndimage as scn
import pandas as pd
pd.options.display.max_rows = 1000
from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
contours_file = open(sys.argv[1])
coords_file   = pd.read_csv(sys.argv[2],index_col = 0)

wsi_object    = WholeSlideImage(sys.argv[3])
contours      = geojson.load(contours_file)
frac_pos_list = []
nb = 0
print(coords_file.shape)
for x,y in zip(coords_file["coords_x"],coords_file["coords_y"]):
    for n in range(len(contours)):
        if(nb%1000==0): print(nb)
        nb = nb+1
        #if(nb>1000): continue
        ##Visualize 
        """
        vis_level = 3                
        dim = wsi_object.level_dim[vis_level]
        points = np.array(contours[n]['geometry']['coordinates']).squeeze()
        points_scaled = np.int32(points/wsi_object.wsi.level_downsamples[vis_level])
        x_scaled, y_scaled = x/wsi_object.wsi.level_downsamples[vis_level], y/wsi_object.wsi.level_downsamples[vis_level]
        img = np.array(wsi_object.wsi.read_region((0,0),vis_level,dim).convert("RGB"))

        plt.plot(points_scaled[:,0],points_scaled[:,1])
        plt.plot([x_scaled],[y_scaled],'ro')
        plt.imshow(img)
        plt.show()
        """
        ##Now extract
        vis_level = 0
        dim = (256,256)
        points = np.array(contours[n]['geometry']['coordinates'])
        points_scaled = np.int32(points/wsi_object.wsi.level_downsamples[vis_level])

        mask = np.zeros((dim[0],dim[1]))
        point_temp = points_scaled - [x,y] ## shift the contour to be centered on the image        
        cv2.fillConvexPoly(mask, np.int32(point_temp), (1)) ## Create the mask
        #if(np.max(mask)>0 and np.min(mask)==0.0):

        img = np.array(wsi_object.wsi.read_region((x,y),vis_level,dim).convert("RGB"))
        #plt.imshow(img)
        #plt.imshow(mask,alpha=0.5)
        #plt.show()
        id_pos = np.where(mask.flatten()==1)
        id_neg = np.where(mask.flatten()==0)
        frac_pos = float(len(id_pos[0]))/len(mask.flatten())
        frac_pos_list.append(frac_pos)


coords_file['tumour_label'] = pd.Series(frac_pos_list,coords_file.index[:len(frac_pos_list)])
coords_file.to_csv(sys.argv[2][:-4]+"_labelled.csv")

