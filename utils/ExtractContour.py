import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csc_matrix
import scipy.ndimage as scn

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson

contours_file = open(sys.argv[1])
coords_file = 
wsi_object = WholeSlideImage(sys.argv[3])
contours = geojson.load(contours_file)
coords = coords_file['coords']
frac_pos_list = []
frac_neg_list = []
for n in range(len(contours)):
    vis_level = 0        
    dim = (512,512)
    points = np.array(contours[n]['geometry']['coordinates'])
    points = np.int32(points/wsi_object.wsi.level_downsamples[vis_level])
    nb = 0
    for xmin,ymin in coords:
        mask = np.zeros((dim[1],dim[1]))
        point_temp = points - [xmin,ymin] ## shift the contour to be centered on the image        
        cv2.fillConvexPoly(mask, np.int32(point_temp), (1)) ## Create the mask
        #if(np.max(mask)>0 and np.min(mask)==0.0):
        
        id_pos = np.where(mask.flatten()==1)
        id_neg = np.where(mask.flatten()==0)
        frac_pos = float(len(id_pos[0]))/len(mask.flatten())
        frac_neg = float(len(id_neg[0]))/len(mask.flatten())
        frac_pos_list.append(frac_pos)
        frac_neg_list.append(frac_neg)
        img = np.array(wsi_object.wsi.read_region((xmin,ymin),vis_level,dim))[:,:,:3]
        """
        #Visualize
        plt.imshow(img)
        plt.show()
        plt.imshow(img)            
        plt.imshow(mask, alpha=0.5)
        plt.show()
        """
        np.savez_compressed('Output/'+str(xmin)+"_"+str(ymin), img=img, mask=mask) 




