import numpy as np
import openslide
from matplotlib import pyplot as plt
import os
import torch
import ColourNorm
import staintools
from skimage.transform import resize
from sys import platform
import seaborn as sns
import matplotlib

if platform == 'darwin':
    bp0 = '/Users/mikael'
else:
    bp0 = '/home/dgs'
    matplotlib.use('TkAgg')

# Select training slide
id_train = '484813'
bp = "Dropbox/M/PostDoc/UCL/datasets/Digital_Pathology/zhuoyan/ColourNorm/"

# Select testing slide
bp_test = "Dropbox/M/PostDoc/UCL/datasets/Digital_Pathology/zhuoyan/ColourNorm/"
# Some slides we have tested: RNOH_S00104604_164604, RNOH_S00104812_165047, RNOH_S00104628_164811, 492090, 492007, 492040, 499877, 493199.
id_test = 'RNOH_S00104812_165047'

# --------------------------------------------------------------------------------------------------------------------
# Fixed parameters of Macenko normalisation - see class for more details.
alpha = 1
beta = 0.1
Io = 250
export = False

# Visibility level for processing. LEAVE TO ZERO, otherwise colour norm will be biased to a zoom level.
vis = 0

# Load training image and select adequate patch of image.
START = (50000, 31000)
SIZE = (15000, 15000)
wsi_object_train = openslide.open_slide(os.path.join(bp0, bp, id_train + '.svs'))
img_train = np.array(wsi_object_train.read_region(START, vis, SIZE).convert("RGB"))

# Train
print('Starting to train....')
MacenkoNormaliser = ColourNorm.Macenko(alpha=alpha, beta=beta, Io=Io)
img_train_for_norm = torch.from_numpy(img_train).permute(2, 0, 1)  # torch of size C x H x W.
MacenkoNormaliser.fit(img_train_for_norm)
print('Training completed.')

# Also train with staintools (other normaliser from literature, to compare with our implementation)
print('Starting to train with staintools....')
MacenkoNormaliser_Staintools = staintools.StainNormalizer(method="Macenko")
MacenkoNormaliser_Staintools.fit(img_train)
print('Training completed.')

# --------------------------------------------------------------------------------------------------------------------
# Fit multiple test slides

wsi_object_test = openslide.open_slide(os.path.join(bp0, bp_test, id_test + '.svs'))

# Relevant patches for WSI ID# 493199
# start2 = (21477, 33909) # fat
# size2=(256, 256)
# start3 = (0, 0)  # background fail
# size3 = (2048, 2048)
# start4 = (20224+256,47360)  # tumour
# size4 = (256, 256)
# start5 = (24000, 46200)  # tumour
# size5 = (512, 512)

# Relevant patches for WSI ID# 493199
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

# Relevant patches for WSI ID# RNOH_S00104604_164604.svs (examples of MF)
# start2 = (67000, 26700)
# size2 = (2048, 2048)
# start3 = (64000, 25000)
# size3 = (2048, 2048)
# start4 = (126666, 71645)
# size4 = (2048, 2048)

# Relevant patches for WSI ID# RNOH_S00104604_164811.svs (examples of MF)
# start2 = (94500, 30100)
# size2 = (2048, 2048)
# start3 = (13500, 45200)
# size3 = (2048, 2048)
# start4 = (50255, 22530)
# size4 = (2048, 2048)

# Relevant patches for WSI ID# RNOH_S00104812_165047 (examples of MF)
start2 = (26000, 41000)
size2 = (256, 256)
start3 = (26700, 33000)
size3 = (2048, 2048)
start4 = (31600, 42000)
size4 = (1024, 1024)

# Relevant patches for WSI ID# 492040 (examples of MF)
# start2 = (60000, 27000)
# size2 = (2048, 2048)
# start3 = (64100, 48300)
# size3 = (2048, 2048)
# start4 = (57400, 66000)
# size4 = (2048, 2048)


# Relevant patches for WSI ID# 492007 (examples of MF)
# start2 = (105000, 37600)
# size2 = (2048, 2048)
# start3 = (15400, 40000)
# size3 = (2048, 2048)
# start4 = (94300, 40000)
# size4 = (2048, 2048)


# Relevant patches for WSI ID# 492090 (no MFs)
# start2 = (102000, 62000)
# size2 = (2048, 2048)
# start3 = (19000, 62000)
# size3 = (2048, 2048)
# start4 = (48500, 25280)
# size4 = (2048, 2048)

# Loop over each region
starts = [start2, start3, start4]
sizes = [size2, size3, size4]

for start, size in zip(starts, sizes):
    print('Processing ROI ({},{})...'.format(start[0], start[1]))

    img_test = np.array(wsi_object_test.read_region(start, vis, size).convert("RGB"))

    # Fit - our implementation
    img_test_for_norm = torch.from_numpy(img_test).permute(2, 0, 1)  # torch of size C x H x W.
    img_test_norm, H, E = MacenkoNormaliser.normalize(img_test_for_norm, stains=True)

    # Fit - comparison with staintools
    img_test_norm_Staintools = MacenkoNormaliser_Staintools.transform(img_test)

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.imshow(resize(img_train, (512, 512)))  # reshape for display because the training ROI is too large.
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

    plt.subplot(2, 3, 6)
    plt.imshow(img_test_norm_Staintools)
    plt.title('Normalized with staintools')

    # Histograms
    # sns.set_style("darkgrid")
    # sns.set_context('talk', font_scale=1.2)
    # fig, ax = plt.subplot_mosaic("ABC", figsize=(17, 4))
    # Rn, Gn, Bn = norm_test_img[:,:,0].flatten(), norm_test_img[:,:,1].flatten(), norm_test_img[:,:,2].flatten()
    # R, G, B = img_test[:, :, 0].flatten(), img_test[:, :, 1].flatten(), img_test[:, :, 2].flatten()
    # bins = np.linspace(0, 255, 30)
    # Ap = 0.7
    # dens = True
    # ax['A'].hist(R, bins=bins, alpha=Ap, color='darkred', density=dens)
    # ax['A'].hist(Rn, bins=bins, alpha=Ap, color='lightcoral', density=dens)
    # ax['A'].legend(['Raw','Norm'])
    # ax['B'].set_title('R')
    #
    # ax['B'].hist(G, bins=bins, alpha=Ap, color='darkgreen', density=dens)
    # ax['B'].hist(Gn, bins=bins, alpha=Ap, color='springgreen', density=dens)
    # ax['B'].legend(['Raw','Norm'])
    # ax['B'].set_title('G')
    #
    # ax['C'].hist(B, bins=bins, alpha=Ap, color='darkblue', density=dens)
    # ax['C'].hist(Bn, bins=bins, alpha=Ap, color='cornflowerblue', density=dens)
    # ax['C'].legend(['Raw','Norm'])
    # ax['C'].set_title('B')


# Export the current colour calibration.
if export:

    filename = './trained/' + id_train + '_vis' + str(vis) + '_HERef.pt'
    HEref = MacenkoNormaliser.HERef
    maxCRef = MacenkoNormaliser.maxCRef
    alpha = MacenkoNormaliser.alpha
    beta = MacenkoNormaliser.beta
    Io = MacenkoNormaliser.Io
    torch.save({'HERef': HEref, 'maxCRef': maxCRef, 'alpha': alpha, 'beta': beta, 'Io': Io}, filename)