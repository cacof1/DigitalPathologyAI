# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
# other imports
from pathlib import Path
import os, glob
import time
import argparse

if __name__ == '__main__':

    ## Parser
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--source', type=str, help='path to folder containing raw wsi image files')
    parser.add_argument('--step_size', type=int, default=256, help='step_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--patch', default=True, action='store_true')
    parser.add_argument('--seg', default=True, action='store_true')
    parser.add_argument('--stitch', default=True, action='store_true')
    parser.add_argument('--no_auto_skip', default=True, action='store_false')
    parser.add_argument('--save_dir', type=str, help='directory to save processed data')
    parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')

    args = parser.parse_args(['--source', '/media/mikael/LaCie/test/',#'/media/mikael/LaCie/sarcoma/svs/',
                              '--save_dir', '/media/mikael/LaCie/test/',#'/media/mikael/LaCie/sarcoma/',
                              '--patch_size', '256',
                              '--step_size', '256',
                              ])

    config = vars(args)
    config["patch_save_dir"] = os.path.join(args.save_dir, 'patches')
    config["mask_save_dir"] = os.path.join(args.save_dir, 'QA', 'masks')
    config["stitch_save_dir"] = os.path.join(args.save_dir, 'QA', 'stitches')
    ### End of parser

    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': config['patch_save_dir'],
                   'mask_save_dir': config['mask_save_dir'],
                   'stitch_save_dir': config['stitch_save_dir']}

    for key, val in directories.items():  ## Create directories
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    source = Path(args.source).glob("**/*.svs")
    for nb, slide in enumerate(source):
        seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none',
                      'exclude_ids': 'none'}
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
        vis_params = {'vis_level': -1, 'line_thickness': 250}
        patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

        seg_times = 0
        patch_times = 0.
        stitch_times = 0.

        print('processing {}'.format(slide.stem), nb)
        slide_id = slide.stem

        # Initialize WSI
        WSI_object = WholeSlideImage(str(slide))

        ## Find best vis/seg level
        if vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                vis_params['vis_level'] = 0

            else:
                best_level = WSI_object.wsi.get_best_level_for_downsample(64)
                if best_level >= len(WSI_object.level_dim):
                    vis_params['vis_level'] = len(WSI_object.level_dim) - 1
                else:
                    vis_params['vis_level'] = best_level

        if seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                seg_params['seg_level'] = 0

            else:
                best_level = WSI_object.wsi.get_best_level_for_downsample(64)
                if best_level >= len(WSI_object.level_dim):
                    seg_params['seg_level'] = len(WSI_object.level_dim) - 1
                else:
                    seg_params['seg_level'] = best_level

        w, h = WSI_object.level_dim[seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            continue

        config.update(seg_params)
        config.update(filter_params)
        config.update(patch_params)
        config.update(vis_params)

        ## Segmentation
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
        mask = WSI_object.visWSI(**vis_params)
        mask_path = os.path.join(config['mask_save_dir'], slide_id + '.jpg')
        mask.save(mask_path)

        ## Patching
        patch_params.update(
            {'patch_level': args.patch_level, 'patch_size': args.patch_size, 'step_size': args.step_size,
             'save_path': config['patch_save_dir']})
        dataframe = WSI_object.process_contours(**patch_params)

        ## Stitching
        # stitchmap = StitchCoords(dataframe, config, WSI_object, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False)
        # stitchmap.save(os.path.join(config['stitch_save_dir'], slide_id+'.jpg'))