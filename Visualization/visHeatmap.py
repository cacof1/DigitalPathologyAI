import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import cv2
from PIL import Image


def _assertLevelDownsamples(wsi_object):
    level_downsamples = []
    dim_0 = wsi_object.level_dimensions[0]

    for downsample, dim in zip(wsi_object.level_downsamples, wsi_object.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
        downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def to_percentiles(scores):

    scores = rankdata(scores, 'average')/len(scores) * 100
    return scores

def block_blending(wsi_object, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
    print('\ncomputing blend')
    level_downsamples = _assertLevelDownsamples(wsi_object)
    downsample = level_downsamples[vis_level]
    w = img.shape[1]
    h = img.shape[0]
    block_size_x = min(block_size, w)
    block_size_y = min(block_size, h)
    print('using block size: {} x {}'.format(block_size_x, block_size_y))

    shift = top_left  # amount shifted w.r.t. (0,0)
    for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
        for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
            # print(x_start, y_start)

            # 1. convert wsi coordinates to image coordinates via shift and scale
            x_start_img = int((x_start - shift[0]) / int(downsample[0]))
            y_start_img = int((y_start - shift[1]) / int(downsample[1]))

            # 2. compute end points of blend tile, careful not to go over the edge of the image
            y_end_img = min(h, y_start_img + block_size_y)
            x_end_img = min(w, x_start_img + block_size_x)

            if y_end_img == y_start_img or x_end_img == x_start_img:
                continue
            # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

            # 3. fetch blend block and size
            blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
            blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

            if not blank_canvas:
                # 4. read actual wsi block as canvas block
                pt = (x_start, y_start)
                canvas = np.array(wsi_object.read_region(pt, vis_level, blend_block_size).convert("RGB"))
            else:
                # 4. OR create blank canvas block
                canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

            # 5. blend color block and canvas block
            img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                1 - alpha, 0, canvas)
    return img

def visHeatmap(wsi_object, scores, coords, vis_level=-1,
               top_left=None, bot_right=None,
               patch_size=(256, 256),
               blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
               blur=False, overlap=0.0,
               segment=True, use_holes=True,
               convert_to_percentiles=False,
               binarize=False, thresh=0.5,
               max_size=None,
               custom_downsample=1,
               cmap='coolwarm'):
    """
    Args:
        scores (numpy array of float): Attention scores
        coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
        vis_level (int): WSI pyramid level to visualize
        patch_size (tuple of int): Patch dimensions (relative to lvl 0)
        blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
        canvas_color (tuple of uint8): Canvas color
        alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
        blur (bool): apply gaussian blurring
        overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
        use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
        convert_to_percentiles (bool): whether to convert attention scores to percentiles
        binarize (bool): only display patches > threshold
        thresh (float): binarization threshold
        max_size (int): Maximum canvas size (clip if goes over)
        custom_downsample (int): additionally downscale the heatmap by specified factor
        cmap (str): name of matplotlib colormap to use
    """

    if vis_level < 0:
        vis_level = wsi_object.get_best_level_for_downsample(32)

    level_downsamples = _assertLevelDownsamples(wsi_object)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level

    if len(scores.shape) == 2:
        scores = scores.flatten()

    if binarize:
        if thresh < 0:
            threshold = 1.0 / len(scores)

        else:
            threshold = thresh

    else:
        threshold = 0.0

    ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)

    else:
        region_size = wsi_object.level_dimensions[vis_level]
        top_left = (0, 0)
        bot_right = wsi_object.level_dimensions[0]
        w, h = region_size

    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    print('\ncreating heatmap for: ')
    print('top_left: ', top_left, 'bot_right: ', bot_right)
    print('w: {}, h: {}'.format(w, h))
    print('scaled patch size: ', patch_size)

    ###### normalize filtered scores ######
    if convert_to_percentiles:
        scores = to_percentiles(scores)

    #scores /= 100

    ######## calculate the heatmap of raw attention scores (before colormap)
    # by accumulating scores over overlapped regions ######

    # heatmap overlay: tracks attention score over each pixel of heatmap
    # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    overlay = np.full(np.flip(region_size), 0).astype(float)
    counter = np.full(np.flip(region_size), 0).astype(np.uint16)
    count = 0
    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            if binarize:
                score = 1.0
                count += 1
        else:
            score = 0.0
        # accumulate attention
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

    if binarize:
        print('\nbinarized tiles based on cutoff of {}'.format(threshold))
        print('identified {}/{} patches as positive'.format(count, len(coords)))

    # fetch attended region and average accumulated attention
    zero_mask = counter == 0

    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter
    if blur:
        overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if not blank_canvas:
        # downsample original image and use as canvas
        img = np.array(wsi_object.read_region(top_left, vis_level, region_size).convert("RGB"))
    else:
        # use blank canvas
        img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        # return Image.fromarray(img) #raw image

    print('\ncomputing heatmap image')
    print('total of {} patches'.format(len(coords)))
    twenty_percent_chunk = max(1, int(len(coords) * 0.2))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    for idx in range(len(coords)):
        if (idx + 1) % twenty_percent_chunk == 0:
            print('progress: {}/{}'.format(idx, len(coords)))

        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:

            # attention block
            raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]

            # image block (either blank canvas or orig image)
            img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()

            # color block (cmap applied to attention block)
            color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

            img_block = color_block

            # rewrite image block
            img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()

    # return Image.fromarray(img) #overlay
    print('Done')
    del overlay

    if blur:
        img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if alpha < 1.0:
        img = block_blending(wsi_object,
                             img, vis_level, top_left, bot_right,
                             alpha=alpha, blank_canvas=blank_canvas,
                             block_size=1024)

    img = Image.fromarray(img)
    w, h = img.size

    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img