import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csc_matrix
import scipy.ndimage as scn
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
df = pd.DataFrame()
contours_file = open(sys.argv[1])
coords_file = h5py.File(sys.argv[2], "r")
wsi_object = WholeSlideImage(sys.argv[3])
contours = geojson.load(contours_file)
coords = coords_file['coords']
frac_pos_list = []
frac_neg_list = []
vis_level = 0        
dim = (512,512)
polygon_points = np.array(contours[0]['geometry']['coordinates'])
polygon_points = np.int32(polygon_points/wsi_object.wsi.level_downsamples[vis_level])
nb = 0
polygon = Polygon(polygon_points)
nb = 0
for xmin,ymin in coords:
    point = Point(xmin,ymin)
    img = np.array(wsi_object.wsi.read_region((xmin,ymin),vis_level,dim))[:,:,:3]
    np.savez_compressed('Output/'+str(xmin)+"_"+str(ymin), img=img, value=polygon.contains(point))
    if(nb%1000==0) : print(nb)

