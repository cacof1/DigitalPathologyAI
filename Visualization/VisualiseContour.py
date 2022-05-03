import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import openslide
import geojson

with open(sys.argv[1]) as f: data = geojson.load(f)

nContours = len(data)
print(nContours)

##Just our selection
dataset = np.load(sys.argv[3])
coord_x, coord_y = map(int,sys.argv[3][:-4].split("/")[-1].split("_"))

## Full slide with contour
points = np.array(data[0]['geometry']['coordinates'])
wsi_object = openslide.open_slide(sys.argv[2])
wsi_object.associated_images['label'].save('test.png')
vis_level = 2
dim = wsi_object.level_dimensions[vis_level]
points = np.int32(points/wsi_object.level_downsamples[vis_level])
img = np.array(wsi_object.read_region((0,0),vis_level,dim))
mask = np.zeros((dim[1],dim[0]))
cv2.fillConvexPoly(mask, np.int32(points), (255,255,255,255))
downsample = int(wsi_object.level_downsamples[vis_level])
coord_x  /= downsample
coord_y  /= downsample
dim_patch = np.round(512/downsample).astype(np.int32)

plt.imshow(img,origin='lower')
plt.imshow(mask,alpha=0.5,origin='lower')
plt.plot([coord_x, coord_x+dim_patch, coord_x+dim_patch,coord_x,coord_x],[coord_y,coord_y,coord_y+dim_patch,coord_y+dim_patch,coord_y],'r-')
plt.show()

plt.imshow(dataset['img'],origin='lower')
plt.imshow(dataset['mask'],alpha=0.5,origin='lower')
plt.show()



