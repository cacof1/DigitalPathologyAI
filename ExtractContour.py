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
coords_file = h5py.File(sys.argv[2], "r")
wsi_object = WholeSlideImage(sys.argv[3])
contours = geojson.load(contours_file)
coords = coords_file['coords']

for n in range(len(contours)):
    vis_level = 0        
    dim = (256,256)
    points = np.array(contours[n]['geometry']['coordinates'])
    points = np.int32(points/wsi_object.wsi.level_downsamples[vis_level])
    nb = 0
    for xmin,ymin in coords:
        nb = nb+1
        if(nb<15000): continue
        mask = np.zeros((dim[1],dim[1]))
        point_temp = points - [xmin,ymin] ## shift the contour to be centered on the image
        cv2.fillConvexPoly(mask, np.int32(point_temp), (1)) ## Create the mask
        if(np.max(mask)>0 and np.min(mask)==0.0):
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




