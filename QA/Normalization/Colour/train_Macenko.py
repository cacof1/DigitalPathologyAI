import torch.nn as nn
import torch
from typing import Union
import torch.nn.functional as F
import ColourNorm
import numpy as np
from wsi_core.WholeSlideImage import WholeSlideImage
from matplotlib import pyplot as plt

id_test = '492006'
id_train = '484813'
#id_train = '493199'
#id_test = '493199'
bp='/Users/mikael/Documents/sarcoma/'
bp = '/media/mikael/LaCie/sarcoma/'

# Params of Macenko
alpha = 1
beta = 0.1
Io = 250

# for viewing
vis = 0
start = (0, 0)

patient_id_train = bp + id_train + '.svs'
wsi_object_train = WholeSlideImage(patient_id_train)
start = (int(wsi_object_train.level_dim[vis][0]/3), int(wsi_object_train.level_dim[vis][1]/3))
#size = wsi_object_train.level_dim[vis]
size = (2000, 2000)
img_train = np.array(wsi_object_train.wsi.read_region(start, vis, size).convert("RGB"))

patient_id_test = bp + id_test + '.svs'
wsi_object_test = WholeSlideImage(patient_id_test)
start = (int(wsi_object_test.level_dim[vis][0]/3), int(wsi_object_test.level_dim[vis][1]/3))
size = (256, 256)
img_test = np.array(wsi_object_test.wsi.read_region(start, vis, size).convert("RGB"))

# Train
MacenkoNormaliser = ColourNorm.Macenko(alpha=alpha, beta=beta, Io=Io)
img_train_for_norm = torch.from_numpy(img_train).permute(2, 0, 1)  # torch of size C x H x W.
MacenkoNormaliser.fit(img_train_for_norm)

# Fit
img_test_for_norm = torch.from_numpy(img_test).permute(2, 0, 1)  # torch of size C x H x W.
img_test_norm, H, E = MacenkoNormaliser.normalize(img_test_for_norm, stains=True)

plt.subplot(2, 3, 1)
plt.imshow(img_train)
plt.title('Reference image')

plt.subplot(2, 3, 2)
plt.imshow(img_test)
plt.title('Original test image')

plt.subplot(2, 3, 3)
plt.imshow(img_test_norm.cpu().detach().numpy())
plt.title('Normalized test image')

plt.subplot(2, 3, 4)
plt.imshow(H)
plt.title('Haematoxylin channel')

plt.subplot(2, 3, 5)
plt.imshow(E)
plt.title('Eosin channel')
plt.show()

# Export the current model!

filename = './trained/' + id_train + '_vis' + str(vis) + '_HERef.pt'
HEref = MacenkoNormaliser.HERef
maxCRef = MacenkoNormaliser.maxCRef
alpha = MacenkoNormaliser.alpha
beta = MacenkoNormaliser.beta
Io = MacenkoNormaliser.Io
torch.save({'HERef': HEref, 'maxCRef': maxCRef, 'alpha': alpha, 'beta': beta, 'Io': Io}, filename)

## Sanity check
#MacenkoNormaliser2 = ColourNorm.Macenko(saved_fit_file=filename)
#img_test_norm2, H2, E2 = MacenkoNormaliser2.normalize(img_test_for_norm, stains=True)
#plt.imshow(img_test_norm2)
#plt.show()


## Classify Zhuoyan's datasets
he = np.load('./test_tiles/h_e_tile.npy')
phh3 = np.load('./test_tiles/phh3_tile.npy')

MacenkoNorm = ColourNorm.Macenko(saved_fit_file=filename)

tile_test = torch.from_numpy(he).permute(2, 0, 1)  # torch of size C x H x W.
tile_test_norm, Htile, Etile = MacenkoNorm.normalize(tile_test, stains=True)

plt.figure()

plt.subplot(2, 3, 1)
plt.imshow(he)
plt.title('Base image')

plt.subplot(2, 3, 2)
plt.imshow(tile_test_norm)
plt.title('Normalized image')

plt.subplot(2, 3, 3)
plt.imshow(phh3)
plt.title('PHH3 image')

plt.subplot(2, 3, 4)
plt.imshow(Htile)
plt.title('Haematoxylin channel')

plt.subplot(2, 3, 5)
plt.imshow(Etile)
plt.title('Eosin channel')
plt.show()






