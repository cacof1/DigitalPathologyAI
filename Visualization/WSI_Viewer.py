import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def get_downsample(WSI_object, vis_level):
    dim_0 = WSI_object.level_dimensions[0]
    dim = WSI_object.level_dimensions[vis_level]
    estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))

    return estimated_downsample


def block_blending(img, WSI_object, vis_level, top_left, bot_right, alpha=0.5, block_size=1024):
    downsample = get_downsample(WSI_object, vis_level)
    w = img.shape[1]
    h = img.shape[0]
    block_size_x = min(block_size, w)
    block_size_y = min(block_size, h)
    # print('using block size: {} x {}'.format(block_size_x, block_size_y))

    shift = top_left  # amount shifted w.r.t. (0,0)
    for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
        for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):

            # 1. convert wsi coordinates to image coordinates via shift and scale
            x_start_img = int((x_start - shift[0]) / int(downsample[0]))
            y_start_img = int((y_start - shift[1]) / int(downsample[1]))

            # 2. compute end points of blend tile, careful not to go over the edge of the image
            y_end_img = min(h, y_start_img + block_size_y)
            x_end_img = min(w, x_start_img + block_size_x)

            if y_end_img == y_start_img or x_end_img == x_start_img:
                continue

            # 3. fetch blend block and size
            blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
            blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

            # 4. read actual wsi block as canvas block
            pt = (x_start, y_start)
            canvas = np.array(WSI_object.read_region(pt, vis_level, blend_block_size).convert("RGB"))

            # 5. blend color block and canvas block
            img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                1 - alpha, 0, canvas)
    return img


def generate_overlay(WSI_object=None, labels=None, coords=None, vis_level=2,
                     patch_size=[256, 256], cmap=None, alpha=0.4):
    # INPUT:
    # WSI_object: an instance of the class openslide, ex: WSI_object = openslide.open_slide(svs_filename).
    # labels: a (Ncoords, ) numpy array providing the label of each coordinate at coords
    # coords: a (Ncoords, 2) numpy array with (x,y) starting points of relevant patches
    # vis_level: visibility level to generate the overlay. See openslide for more info.
    # patch_size: list providing the original patch size
    # cmap: either a string pointing to a pyplot colormap name, or a matplotlib.colors.ListedColormap object.

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Get scaling factor
    downsample = get_downsample(WSI_object, vis_level)
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level

    # Get region size, locations
    w, h = WSI_object.level_dimensions[vis_level]
    top_left = (0, 0)
    bot_right = WSI_object.level_dimensions[0]

    scaled_patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    scaled_coords = np.ceil(coords * np.array(scale)).astype(int)

    # heatmap overlay: tracks attention score over each pixel of heatmap
    # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    overlay = np.full((h, w), 0).astype(float)
    counter = np.full((h, w), 0).astype(np.uint16)
    count = 0

    for idx in range(len(scaled_coords)):
        label = labels[idx]
        coord = scaled_coords[idx]

        # accumulate attention
        overlay[coord[1]:coord[1] + scaled_patch_size[1], coord[0]:coord[0] + scaled_patch_size[0]] += label
        # accumulate counter
        counter[coord[1]:coord[1] + scaled_patch_size[1], coord[0]:coord[0] + scaled_patch_size[0]] += 1

    zero_mask = counter == 0
    overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter

    # downsample original image and use as canvas
    img = np.array(WSI_object.read_region(top_left, vis_level, (w, h)).convert("RGB"))

    twenty_percent_mark = max(1, int(len(scaled_coords) * 0.2))

    for idx in range(len(scaled_coords)):
        # if (idx + 1) % twenty_percent_mark == 0:
        # print('progress: {}/{}'.format(idx, len(coords)))  # uncomment if you want to track image formation

        coord = scaled_coords[idx]

        # Get the "color block", i.e. get the label from the overlay and obtain its colour from colormap
        raw_block = overlay[coord[1]:coord[1] + scaled_patch_size[1], coord[0]:coord[0] + scaled_patch_size[0]]
        rb = raw_block.astype(int)
        color_block = (cmap(rb - 1) * 255)[:, :, :3].astype(np.uint8)

        # Rewrite image block using this colour
        img[coord[1]:coord[1] + scaled_patch_size[1], coord[0]:coord[0] + scaled_patch_size[0]] = color_block.copy()

    # Block blending
    if alpha < 1.0:
        img = block_blending(img, WSI_object, vis_level, top_left, bot_right, alpha=alpha, block_size=1024)

    heatmap = Image.fromarray(img)

    return heatmap, overlay
