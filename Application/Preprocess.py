# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from pathlib import Path
# other imports
import os,glob
import numpy as np
import time
import argparse
import pdb
import pandas as pd

def stitching(file_path, wsi_object, downscale = 64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)


    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                                  patch_size = 256, step_size = 256,
                                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                                  'keep_ids': 'none', 'exclude_ids': 'none'},
                                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
                                  vis_params = {'vis_level': -1, 'line_thickness': 500},
                                  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
                                  patch_level = 0,
                                  use_default_params = False,
                                  seg = False, save_mask = True,
                                  stitch= False,
                                  patch = False, auto_skip=True, process_list = None):


    source    = Path(source).glob("*.svs")
    seg_times = 0
    patch_times = 0.
    stitch_times = 0.
    
    for slide in source:
        print('processing {}'.format(slide.stem))
        slide_id = slide.stem
        
        # Inialize WSI
        WSI_object = WholeSlideImage(str(slide))

        if vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                vis_params['vis_level'] = best_level

        if seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                seg_params['seg_level'] = best_level

        keep_ids = str(seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = seg_params['keep_ids']
            seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else: seg_params['keep_ids'] = []

            
        exclude_ids = str(seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = seg_params['exclude_ids']
            seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else: seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            continue
        

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, seg_params, filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1 # Default time
        if patch:
            patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                                                     'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **patch_params,)

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        #df.loc[idx, 'status'] = 'processed'
        
        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    #df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str, help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256, help='step_size')
parser.add_argument('--patch_size', type = int, default=256, help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str, help='directory to save processed data')
parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None, help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()
    config = vars(args)
    config["patch_save_dir"] = os.path.join(args.save_dir, 'patches')
    config["mask_save_dir"] = os.path.join(args.save_dir, 'masks')
    config["stitch_save_dir"] = os.path.join(args.save_dir, 'stitches')    
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list: process_list = os.path.join(args.save_dir, args.process_list)
    else: process_list = None
    
    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)
    print("hellooooooooooooooo")    
    print("args", config)
    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir' : mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}
    
    for key, val in directories.items(): ## Create directories
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    config.update({'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'}) ## Segmentation
    config.update({'a_t':100, 'a_h': 16, 'max_n_holes':8}) ## Filter
    config.update({'use_padding': True, 'contour_fn': 'four_pt'}) ## Path
    config.update({'vis_level': -1, 'line_thickness': 250}) ## Vis
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                           patch_size = args.patch_size, step_size=args.step_size,
                                           seg = args.seg,  use_default_params=False, save_mask = True,
                                           stitch= args.stitch,
                                           patch_level=args.patch_level, patch = args.patch,
                                           process_list = process_list, auto_skip=args.no_auto_skip)
    
