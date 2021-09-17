import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from wsi_core.WholeSlideImage import WholeSlideImage
import geojson

FileList = glob.glob(sys.argv[1]+"/wsi/*.svs")
for filename in filelist:
    filename = filename[:-4]
    ## Load slide
    wsi_object = WholeSlideImage(sys.argv[1]+"/wsi/"+filename+".wsi")
    vis_level = 0        
    dim = (256,256)

    ##Annotation from Tom
    contours_file = open(sys.argv[1]+"/tumour_contour/"+filename+"_annotations.json")
    contours = geojson.load(contours_file)
    polygon_points = np.array(contours[0]['geometry']['coordinates'])
    polygon_points = np.int32(polygon_points/wsi_object.wsi.level_downsamples[vis_level])
    polygon = Polygon(polygon_points)

    ## Load coorsd from CLAM Get rid of previous attempts
    coords_file = h5py.File(sys.argv[1]+"/patches/"+filename+".h5", "r+")
    try: del coords_file['label']
    except: print('hello')
    coords = coords_file['coords']
    
    nb = 0
    df = pd.DataFrame()
    for xmin,ymin in coords:
        point = Point(xmin,ymin)
        img = np.array(wsi_object.wsi.read_region((xmin,ymin),vis_level,dim))#[:,:,:3]
        values_to_add = {}
        values_to_add["point"] = (xmin,ymin)
        values_to_add["label"] = polygon.contains(point)
        df = df.append(pd.Series(values_to_add,name=nb))
        if(nb%100==0) : print(nb)
        nb = nb+1
    print(df)
    coords_file['label']  = df['label']
    coords_file.close()
