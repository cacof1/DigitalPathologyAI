import numpy as np
import openslide
from matplotlib import pyplot as plt
import os
import torch
import ColourNorm
import staintools

id_test = '499877'
id_test = '493199'
id_train = '484813'
bp = '/Users/mikael/Documents/testing/'

# --------------------------------------------------------------------------------------------------------------------
# hyperparameters of Macenko normalisation
alpha = 1
beta = 0.1
Io = 250
export = False

# for viewing
vis = 0

# Load training image and select adequate patch of image
wsi_object_train = openslide.open_slide(os.path.join(bp, id_train + '.svs'))

# Uncomment to do full image (but RAM will bust)
#start = (int(wsi_object_train.level_dimensions[vis][0]/4), int(wsi_object_train.level_dimensions[vis][1]/4))
#size = wsi_object_train.level_dimensions[vis]

# Current image patch used for normalisation
start = (50000, 31000)
size = (15000, 15000)

# Extract training image patch
img_train = np.array(wsi_object_train.read_region(start, vis, size).convert("RGB"))

# Train
MacenkoNormaliser = ColourNorm.Macenko(alpha=alpha, beta=beta, Io=Io)
img_train_for_norm = torch.from_numpy(img_train).permute(2, 0, 1)  # torch of size C x H x W.
MacenkoNormaliser.fit(img_train_for_norm)

# Also train with staintools (other normaliser from literature, to compare with our implementation)
MacenkoNormaliser_Staintools = staintools.StainNormalizer(method="Macenko")
MacenkoNormaliser_Staintools.fit(img_train)

# --------------------------------------------------------------------------------------------------------------------
# Fit multiple test slides

wsi_object_test = openslide.open_slide(os.path.join(bp, id_test + '.svs'))

# Region with tissue and background
start1 = (int(wsi_object_test.level_dimensions[vis][0]/3), int(wsi_object_test.level_dimensions[vis][1]/3))
size1 = (2048, 2048)

# Example of fat
start2 = (21000, 33000)
size2 = (2048, 2048)

start2 = (21477, 33909)
size2=(256, 256)

# Example on how it fails in the background
start3 = (0, 0)
size3 = (2048, 2048)

# example in tumour
# start4 = (26600, 25000)
# size4 = (4096, 4096)
start4 = (20224+256,47360)
size4 = (256, 256)

# other tumour example
start5 = (24000, 46200)
size5 = (512, 512)

# Some additional interesting examples
# start1 = (21000, 18500)
# start2 = (27000, 23000)
# start3 = (11150, 15120)
# start4 = (13500, 34500)
# start5 = (45789, 13600)
# size1 = (512, 512)
# size2 = (512, 512)
# size3 = (512, 512)
# size4 = (512, 512)
# size5 = (512, 512)

# Loop over each region
starts = [start1, start2, start3, start4, start5]
sizes = [size1, size2, size3, size4, size5]

for start, size in zip(starts, sizes):

    img_test = np.array(wsi_object_test.read_region(start, vis, size).convert("RGB"))

    # Fit - our code
    img_test_for_norm = torch.from_numpy(img_test).permute(2, 0, 1)  # torch of size C x H x W.
    img_test_norm, H, E = MacenkoNormaliser.normalize(img_test_for_norm, stains=True)

    # Fit - comparison with staintools
    img_test_norm_Staintools = MacenkoNormaliser_Staintools.transform(img_test)

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.imshow(img_train)
    plt.title('Reference image')

    plt.subplot(2, 3, 2)
    plt.imshow(img_test)
    plt.title('Original test image')


    plt.subplot(2, 3, 3)
    norm_test_img = img_test_norm.cpu().detach().numpy()
    plt.imshow(norm_test_img)
    plt.title('Normalized test image')

    plt.subplot(2, 3, 4)
    plt.imshow(H)
    plt.title('Haematoxylin channel')

    plt.subplot(2, 3, 5)
    plt.imshow(E)
    plt.title('Eosin channel')
    plt.show()

    plt.subplot(2, 3, 6)
    plt.imshow(img_test_norm_Staintools)
    plt.title('Normalized w staintools')

# Export the current colour calibration.
if export:

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


## Classify other datasets
# he = np.load('./test_tiles/h_e_tile.npy')
# phh3 = np.load('./test_tiles/phh3_tile.npy')
#
# MacenkoNorm = ColourNorm.Macenko(saved_fit_file=filename)
#
# tile_test = torch.from_numpy(he).permute(2, 0, 1)  # torch of size C x H x W.
# tile_test_norm, Htile, Etile = MacenkoNorm.normalize(tile_test, stains=True)
#
# plt.figure()
#
# plt.subplot(2, 3, 1)
# plt.imshow(he)
# plt.title('Base image')
#
# plt.subplot(2, 3, 2)
# plt.imshow(tile_test_norm)
# plt.title('Normalized image')
#
# plt.subplot(2, 3, 3)
# plt.imshow(phh3)
# plt.title('PHH3 image')
#
# plt.subplot(2, 3, 4)
# plt.imshow(Htile)
# plt.title('Haematoxylin channel')
#
# plt.subplot(2, 3, 5)
# plt.imshow(Etile)
# plt.title('Eosin channel')
# plt.show()
#
#
#
#
#
#
