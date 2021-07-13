import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson

with open(sys.argv[1]) as f:
    data = geojson.load(f)

nContours = len(data)
print(nContours)
for n in range(nContours):
    points = np.array(data[n]['geometry']['coordinates'])
    wsi_object = WholeSlideImage(sys.argv[2])
    wsi_object.wsi.associated_images['label'].save('test.png')
    vis_level = 2
    dim = wsi_object.wsi.level_dimensions[vis_level]
    points = np.int32(points/wsi_object.wsi.level_downsamples[vis_level])
    img = np.array(wsi_object.wsi.read_region((0,0),vis_level,dim))
    mask = np.zeros((dim[1],dim[0]))
    cv2.fillConvexPoly(mask, np.int32(points), (255,255,255,255))
    print(mask.shape)
    dim_patch = np.round(256/int(wsi_object.wsi.level_downsamples[vis_level])).astype(np.int32)
    print(dim_patch)
    coord_x, coord_y = 1000,1000
    plt.imshow(img)
    plt.imshow(mask,alpha=0.5)
    plt.plot([coord_x, coord_x+dim_patch, coord_x+dim_patch,coord_x,coord_x],[coord_y,coord_y,coord_y+dim_patch,coord_y+dim_patch,coord_y],'r-')
    plt.show()
    #mask = mask.astype(bool)
    print(np.max(mask))
    #img[~mask] = (0,0,0,0)
    #plt.imshow(img)
    
    #plt.show()


