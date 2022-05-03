import openslide
import glob
import os
import cv2
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils.OmeroTools
from PIL import Image
from tqdm import tqdm
import torch

sys.path.append('../')
from QA.Normalization.Colour import ColourNorm
from mpire import WorkerPool
import toml


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def plot_contour(xmin, xmax, ymin, ymax, colour='k'):
    plt.plot([xmin, xmax], [ymin, ymin], '-' + colour)
    plt.plot([xmin, xmax], [ymax, ymax], '-' + colour)
    plt.plot([xmin, xmin], [ymin, ymax], '-' + colour)
    plt.plot([xmax, xmax], [ymin, ymax], '-' + colour)


def roi_to_points(df):
    # The correct type is polygon.

    for i in range(len(df)):

        # Could add more if useful
        if df['type'][i] == 'rectangle':
            xmin = str(df['X'][i])
            xmax = str(df['X'][i] + df['Width'][i])
            ymin = str(df['Y'][i])
            ymax = str(df['Y'][i] + df['Height'][i])
            fullstr = xmin + ',' + ymin + ' ' + xmax + ',' + ymin + ' ' + xmax + ',' + ymax + ' ' + xmin + ',' + ymax
            # clockwise!
            df['Points'][i] = fullstr

    return df


def split_ROI_points(coords_string):
    coords = np.array([[float(coord_string.split(',')[0]), float(coord_string.split(',')[1])] for coord_string in
                       coords_string.split(' ')])
    return coords


def lims_to_vec(xmin=0, xmax=0, ymin=0, ymax=0, patch_size=0):
    # Create an array containing all relevant tile edges
    edges_x = np.arange(xmin, xmax, patch_size)
    edges_y = np.arange(ymin, ymax, patch_size)
    EX, EY = np.meshgrid(edges_x, edges_y)
    edges_to_test = np.column_stack((EX.flatten(), EY.flatten()))
    return edges_to_test


# Load a Macenko normaliser
colour_normaliser = ColourNorm.Macenko(saved_fit_file=config['DATA']['Colour_Norm_File'])


def tile_membership(dataset, edge):
    # dataset is a tuple: (coords, patch_size, contours_idx_within_ROI, remove_BG, WSI_object, df). This allows use
    # with MPIRE for multiprocessing, which provides a modest speedup. Unpack:
    coords = dataset[0]
    patch_size = dataset[1]
    contours_idx_within_ROI = dataset[2]
    remove_BG = dataset[3]
    WSI_object = dataset[4]
    df = dataset[5]

    # Start by assuming that the patch is within the contour, and remove it if it does not meet
    # a set of conditions.

    if isinstance(df, pd.DataFrame):

        # First: is the patch within the ROI? Test with cv2 for pre-defined contour,
        # or if no contour then do nothing and use the patch.
        patch_outside_ROI = cv2.pointPolygonTest(coords,
                                                 (edge[0] + patch_size / 2,
                                                  edge[1] + patch_size / 2),
                                                 measureDist=False) == -1
        if patch_outside_ROI:
            return False

        # Second: verify that the valid patch is not within any of the ROIs identified
        # as fully inside the current ROI.
        patch_within_other_ROIs = []
        for ii in range(len(contours_idx_within_ROI)):
            cii = contours_idx_within_ROI[ii]
            object_in_ROI_coords = split_ROI_points(df['Points'][cii]).astype(int)
            patch_within_other_ROIs.append(cv2.pointPolygonTest(object_in_ROI_coords,
                                                                (edge[0] + patch_size / 2,
                                                                 edge[1] + patch_size / 2),
                                                                measureDist=False) >= 0)

        if any(patch_within_other_ROIs):
            return False

    # Third condition: verify if the remove background condition is turned on, and apply.
    # This is the code's main bottleneck as we need to load each patch, colour-normalise it, and assess its colour.
    if remove_BG:

        patch = np.array(WSI_object.wsi.read_region(edge, 0, (patch_size, patch_size)).convert("RGB"))
        img_norm, _, _ = colour_normaliser.normalize(torch.tensor(patch).permute(2, 0, 1), stains=False)
        img_norm = img_norm.numpy()
        patch_norm_gray = img_norm[:, :, 0] * 0.2989 + img_norm[:, :, 1] * 0.5870 + img_norm[:, :, 2] * 0.1140
        background_fraction = np.sum(patch_norm_gray > remove_BG * 255) / np.prod(patch_norm_gray.shape)

        if background_fraction > 0.5:
            return False

        # for debugging/validation purposes: uncomment to plot all patches on top of contours.
        # plot_contour(edge[0], edge[0] + patch_size, edge[1], edge[1] + patch_size)

    return True


class PreProcessor:

    def __init__(self, config):

        # LIST OF INPUTS (see wiki for more information)
        # -------------------------------- Required -------------------------------------
        # vis                : visibility level, leave at 0 for improved performance.
        # patch_size         : scalar value (256, 512, etc)
        # patch_path         : folder where .csv files of pre-processed WSI will be saved.
        # patch_path         : if using contours, folder where images of the processed WSI will be saved for QA.
        # svs_path           : parent path of all svs files. Some files can be located within subfolders of svs_path.
        # ids                : list of ids to process. If empty, will do all slices found recursively in "svs_path".
        # label_name         : string with the name of the label to be created. Defaults to "label".
        # -------------------------------- Optional -------------------------------------
        # contour_path       : string where contours are located or will be located, depending on contour_type below.
        #                      If empty, no contours will be used and the .csv will be generated using all tiles.
        # contour_type       : options: 'local', 'omero', ''/false. If omero, then 'omero_login' dict must be provided.
        # specific_contours  : a list of contours that will be used; the rest are not used. If false, uses all contours.
        # contour_mapping    : a list of strings specifying how to use/rename each contour.
        #                      See config file or wiki for more details.
        # omero_login        : dict with fields required to run utils.OmeroTools.download_omero_ROIs.
        #                      Used if contour_type=='omero'.
        # remove_BG          : if set to a scalar value, this will remove, for each contour, all patches
        #                      whose average colour is remove_BG * [255, 255, 255].
        # remove_BG_contours : a list of contours on which to apply remove_BG defined above. By default, the procedure
        #                      is applied to all contours.

        # Assign default values

        if 'Label_Name' in config['DATA']:
            self.label_name = config['DATA']['Label_Name']
        else:
            self.label_name = 'label'

        if 'Colour_Norm_File' in config['DATA']:
            self.colour_normaliser = ColourNorm.Macenko(saved_fit_file=config['DATA']['Colour_Norm_File'])
            self.colour_norm_file = config['DATA']['Colour_Norm_File']
        else:
            self.colour_norm_file = None

        if 'Vis' in config['DATA']:
            self.vis = config['DATA']['Vis']
        else:
            self.vis = 0

        if 'Patch_Size' in config['DATA']:
            self.patch_size = config['DATA']['Patch_Size']
        else:
            self.patch_size = 256

        if 'Type' in config['CONTOURS']:
            self.contour_type = config['CONTOURS']['Type']
        else:
            self.contour_type = ''

        if 'Patch' in config['PATHS']:
            self.patch_path = config['PATHS']['Patch']
        else:
            self.patch_path = None

        if 'Contour' in config['PATHS']:
            self.contour_path = config['PATHS']['Contour']
        else:
            self.contour_path = None

        if 'QA' in config['PATHS']:
            self.QA_path = config['PATHS']['QA']
        else:
            self.QA_path = None

        if 'WSI' in config['PATHS']:
            self.svs_path = config['PATHS']['WSI']
        else:
            self.svs_path = None

        if 'Ids' in config['DATA']:
            self.ids = config['DATA']['Ids']
        else:
            self.ids = None

        if 'Specific_Contours' in config['CONTOURS']:
            self.specific_contours = config['CONTOURS']['Specific_Contours']
        else:
            self.specific_contours = False

        if 'Remove_Contours' in config['CONTOURS']:
            self.remove_contours = [cont.lower() for cont in config['CONTOURS']['Remove_Contours']]
        else:
            self.remove_contours = ['']

        if 'Remove_BG' in config['CONTOURS']:
            self.remove_BG = config['CONTOURS']['Remove_BG']
        else:
            self.remove_BG = False

        if 'Remove_BG_Contours' in config['CONTOURS']:
            self.remove_BG_contours = [item.lower() for item in config['CONTOURS']['Remove_BG_Contours']]
        else:
            self.remove_BG_contours = ''

        if 'Contour_Mapping' in config['CONTOURS']:
            self.contour_mapping = config['CONTOURS']['Contour_Mapping']
        else:
            self.contour_mapping = False

        if 'OMERO' in config:
            self.omero_login = {'host': config['OMERO']['Host'], 'user': config['OMERO']['User'],
                                'pw': config['OMERO']['Pw'],
                                'target_member': config['OMERO']['Target_Member'],
                                'target_group': config['OMERO']['Target_Group'],
                                'ids': config['DATA']['Ids']}
        else:
            self.omero_login = None

    def contour_intersect(self, cnt_ref, cnt_query):
        # Contour is a 2D numpy array of points (npts, 2)
        # Connect each point to the following point to get a line
        # If any of the lines intersect, then break

        for ref_idx in range(len(cnt_ref) - 1):
            # Create reference line_ref with point AB
            A = cnt_ref[ref_idx, :]
            B = cnt_ref[ref_idx + 1, :]

            for query_idx in range(len(cnt_query) - 1):
                # Create query line_query with point CD
                C = cnt_query[query_idx, :]
                D = cnt_query[query_idx + 1, :]

                # Check if line intersect
                if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                    # If true, break loop earlier
                    return True

        return False

    def create_QA_overlay(self, df_export, WSI_object, patch_size, QA_path, ID, all_possible_labels):
        vis_level_view = len(WSI_object.level_dim) - 1  # always the lowest res vis level
        N_classes = len(all_possible_labels)

        if N_classes <= 10:
            cmap = plt.get_cmap('Set1', lut=N_classes)
        elif N_classes > 10 & N_classes <= 20:
            cmap = plt.get_cmap('tab20', lut=N_classes)
        else:
            cmap = plt.get_cmap('Spectral', lut=N_classes)

        cmap.N = N_classes
        cmap.colors = cmap.colors[0:N_classes]

        # For visual aide: show all colors in the bottom left corner.
        legend_coords = np.array([np.arange(0, N_classes * patch_size, patch_size), np.zeros(N_classes)]).T
        legend_label = all_possible_labels
        all_labels = np.concatenate((legend_label, df_export[self.label_name].values + 1), axis=0)
        all_coords = np.concatenate((legend_coords, np.array(df_export[["coords_x", "coords_y"]])), axis=0)


        """## Broken for now -- to fix
        heatmap, overlay = WSI_object.visHeatmap(all_labels,
                                                 all_coords,
                                                 vis_level=vis_level_view,
                                                 patch_size=(patch_size, patch_size),
                                                 segment=False,
                                                 cmap=cmap,
                                                 alpha=0.4,
                                                 blank_canvas=False,
                                                 return_overlay=True)

        # Draw the contours for each label
        heatmap = np.array(heatmap)
        indexes_to_plot = np.unique(overlay[overlay > 0])
        for ii in range(len(indexes_to_plot)):
            im1 = 255 * (overlay == indexes_to_plot[ii]).astype(np.uint8)
            contours, hierarchy = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            color_label = np.argwhere(indexes_to_plot[ii] == all_possible_labels + 1)[0][0]
            cv2.drawContours(heatmap, contours, -1, 255 * cmap.colors[color_label], 3)

        heatmap_PIL = Image.fromarray(heatmap)
        heatmap_PIL.show()

        # Export image to QA_path to evaluate the quality of the pre-processing.
        os.makedirs(QA_path, exist_ok=True)
        img_pth = os.path.join(QA_path, ID + '_patch_' + str(patch_size) + '.pdf')
        heatmap_PIL.save(img_pth, 'pdf')
        """

    def tile_membership(self, edge, coords, patch_size, contours_idx_within_ROI, remove_BW, WSI_object, df):
        # Start by assuming that the patch is within the contour, and remove it if it does not meet
        # a set of conditions.

        if isinstance(df, pd.DataFrame):

            # First: is the patch within the ROI? Test with cv2 for pre-defined contour,
            # or if no contour then do nothing and use the patch.
            patch_outside_ROI = cv2.pointPolygonTest(coords,
                                                     (edge[0] + patch_size / 2,
                                                      edge[1] + patch_size / 2),
                                                     measureDist=False) == -1
            if patch_outside_ROI:
                return False

            # Second: verify that the valid patch is not within any of the ROIs identified
            # as fully inside the current ROI.
            patch_within_other_ROIs = []
            for ii in range(len(contours_idx_within_ROI)):
                cii = contours_idx_within_ROI[ii]
                object_in_ROI_coords = self.split_ROI_points(df['Points'][cii]).astype(int)
                patch_within_other_ROIs.append(cv2.pointPolygonTest(object_in_ROI_coords,
                                                                    (edge[0] + patch_size / 2,
                                                                     edge[1] + patch_size / 2),
                                                                    measureDist=False) >= 0)

            if any(patch_within_other_ROIs):
                return False

        # Another condition: verify if the remove outlier condition is turned on, and apply.
        # This can be a bit slow as we need to load each patch and assess its colour.
        if remove_BW:

            patch = np.array( WSI_object.read_region(edge, 0, (patch_size, patch_size)).convert("RGB"))
            average_colour = np.mean(patch.reshape(patch_size ** 2, 3), axis=0)
            patch_too_white = np.all(average_colour > remove_BW * np.array([255.0, 255.0, 255.0]))

            if patch_too_white:
                return False
            patch_too_dark = np.all(average_colour < remove_BW * np.array([1.0, 1.0, 1.0]))

            if patch_too_dark:
                return False

            # for debugging/validation purposes: uncomment to plot all patches on top of contours.
            # plot_contour(edge[0], edge[0] + patch_size, edge[1], edge[1] + patch_size)

        return True

    def preprocess_WSI(self):

        # Create patch exportation folder and score file names
        csv_save_dir = os.path.join(self.patch_path, 'csv_vis' + str(self.vis) + '_patch' + str(self.patch_size))
        os.makedirs(csv_save_dir, exist_ok=True)

        contour_files = []
        if self.contour_type == 'local' or self.contour_type == 'omero':

            # Create contours first. This will download all contours.
            if self.contour_type == 'omero':
                utils.OmeroTools.download_omero_ROIs(download_path=self.contour_path, **self.omero_login)
                print('--------------------------------------------------------------------------------')

            # Then, load the local contours (same if contour_type is local or omero)
            for ID in self.ids:
                contour_files.append(os.path.join(self.contour_path, ID + '.svs [0]_roi_measurements.csv'))

        # The first thing we want to do is assign labels (numbers) to the names of contours. To do so, we must loop over
        # the existing datasets and find all the unique contour types. Also, if using a subset of all contours through
        # "used_contours", make sure you only preserve the relevant contours.

        contour_names = []
        for contour_file in contour_files:
            df = pd.read_csv(os.path.join(self.contour_path, contour_file))

            for name in df.Text.unique():

                if all([excluded_contour.lower() not in name.lower() for excluded_contour in self.remove_contours]):

                    if self.specific_contours:
                        if name in self.specific_contours:
                            contour_names.append(name.lower())
                    else:
                        contour_names.append(name.lower())

        unique_contour_names = list(set(contour_names))

        print('-----------------------------')
        print('List of all contours:')
        print(unique_contour_names)
        print('-----------------------------')

        # contour_names gives the name of all existing contours. In some cases, you might want to group some contours
        # together, for instance all artifacts together, all fat [in] or [out] together, etc. We create a mapping
        # from contour_names to contour_names_mapped to do so. If contour_mapping = '', skip this step and just process
        # normally each contour.
        if self.contour_mapping:

            # Do some pre-processing to locate keys: remove spaces and use lowercase.
            keys = [ctr.split(':')[0].replace(' ', '').lower() for ctr in self.contour_mapping]
            values = [ctr.split(':')[1] for ctr in self.contour_mapping]
            dict_contours_to_map = dict(zip(keys, values))
            unique_contour_names_mapped = []

            dict_mapped_contours_label = dict(zip(set(values), np.arange(len(set(values)))))
            unique_contour_names_mapped_label = []
            count_catch_contour = 0

            for ctr_nm in list(unique_contour_names):
                ctr_nm = ctr_nm.replace(' ', '').lower()
                loc = [ctr_nm == key for key in dict_contours_to_map.keys()]
                loc_star = [(key.replace('*', '') in ctr_nm) & ('*' in key) for key in dict_contours_to_map.keys()]

                if any(loc):  # if the contour exists in the mapping
                    unique_contour_names_mapped.append(dict_contours_to_map[ctr_nm])
                    unique_contour_names_mapped_label.append(dict_mapped_contours_label[dict_contours_to_map[ctr_nm]])
                elif any(loc_star):  # if the contour exists, but it's in the format contour*
                    dict_key = [k for ki, k in enumerate(dict_contours_to_map.keys()) if loc_star[ki]][0]
                    unique_contour_names_mapped.append(dict_contours_to_map[dict_key])
                    unique_contour_names_mapped_label.append(dict_mapped_contours_label[dict_contours_to_map[dict_key]])
                elif 'remaining' in keys:  # shortcut to assign all other contours to dict_contours_to_map['remaining']
                    unique_contour_names_mapped.append(dict_contours_to_map['remaining'])
                    unique_contour_names_mapped_label.append(
                        dict_mapped_contours_label[dict_contours_to_map['remaining']])
                else:  # otherwise, use the contour, do not modify it.
                    unique_contour_names_mapped.append(ctr_nm)
                    unique_contour_names_mapped_label.append(count_catch_contour + len(set(values)))
                    count_catch_contour += 1

        else:
            unique_contour_names_mapped = unique_contour_names
            unique_contour_names_mapped_label = np.arange(len(unique_contour_names_mapped))

        Full_Contour_Mapping = {'contour_name': unique_contour_names,
                                'mapped_contour_name': unique_contour_names_mapped,
                                'contour_id': unique_contour_names_mapped_label}

        # Loop over each file and extract relevant indexes.
        for idx in range(len(self.ids)):

            # Preallocate some arrays
            coord_x = []
            coord_y = []
            label = []
            ID = self.ids[idx]
            patch_csv_export_filename = os.path.join(csv_save_dir, ID + '.csv')

            #  load contours if existing, or process entire WSI
            search_WSI_query = os.path.join(self.svs_path, '**', ID + '.svs')
            svs_filename = glob.glob(search_WSI_query, recursive=True)[0]  # if file is hidden recursively
            WSI_object = openslide.open_slide(svs_filename)
            if contour_files:
                df = pd.read_csv(contour_files[idx])
                df = roi_to_points(df)  # converts ROIs that are not polygons to "Points" for uniform handling.
            else:
                xmax, ymax = WSI_object.level_dimensions[0]
                xmin, ymin = 0, 0
                df = [1]  # dummy placeholder as we have a single contour equal to the entire image.

            # Loop over each contour and extract patches contained within
            for i in range(len(df)):

                if isinstance(df, pd.DataFrame):
                    ROI_name = df['Text'][i].lower()
                else:
                    ROI_name = 'entire_wsi'

                print('Processing ROI "{}" ({}/{}) of ID "{}": '.format(ROI_name, str(i + 1), str(len(df)), str(ID)),
                      end='')

                if ROI_name == 'entire_wsi':

                    print('Creating patches for entire image, processing.')

                    xmax, ymax = WSI_object.level_dim[0]
                    edges_to_test = lims_to_vec(xmin=0, xmax=xmax, ymin=0, ymax=ymax, patch_size=self.patch_size)
                    coord_x.append(edges_to_test[:, 0])
                    coord_y.append(edges_to_test[:, 1])
                    label.append(np.ones(len(edges_to_test), dtype=int) * -1)  # should only be a single value.

                elif ROI_name not in Full_Contour_Mapping['contour_name']:

                    print('ROI not within selected contours, skipping.')

                else:

                    print('Found contours, processing.')

                    if isinstance(df, pd.DataFrame):
                        coords = split_ROI_points(df['Points'][i]).astype(int)
                        contour_label = [l for s, l in
                                         zip(Full_Contour_Mapping['contour_name'], Full_Contour_Mapping['contour_id'])
                                         if
                                         ROI_name in s]

                        xmin, ymin = np.min(coords, axis=0)
                        xmax, ymax = np.max(coords, axis=0)

                        # To make sure we do not end up with overlapping contours at the end, round xmin, xmax, ymin,
                        # ymax to the nearest multiple of patch_size.
                        xmin = int(np.floor(xmin / self.patch_size) * self.patch_size)
                        ymin = int(np.floor(ymin / self.patch_size) * self.patch_size)
                        xmax = int(np.ceil(xmax / self.patch_size) * self.patch_size)
                        ymax = int(np.ceil(ymax / self.patch_size) * self.patch_size)

                    else:
                        contour_label = [-1]  # -1 will mean, by definition, that there are no contours assigned.

                    # For debugging: uncomment to see each contour, and its bounding box.
                    # plt.figure()
                    # plt.plot(coords[:, 0], coords[:, 1], '-r')
                    # plot_contour(xmin, xmax, ymin, ymax, colour='g')

                    # -------------------------------------------------------------------------------------------
                    # Get the list of all contours that are contained within the current one.
                    other_ROIs_index = np.setdiff1d(np.arange(len(df)), i)
                    contours_idx_within_ROI = []
                    for jj in range(len(other_ROIs_index)):
                        test_coords = split_ROI_points(df['Points'][other_ROIs_index[jj]]).astype(int)
                        centroid = np.mean(test_coords, axis=0)
                        left_to_centroid = np.vstack([np.array((xmin, centroid[1])), centroid])
                        centroid_to_right = np.vstack([np.array((xmax, centroid[1])), centroid])
                        bottom_to_centroid = np.vstack([np.array((centroid[0], ymin)), centroid])
                        centroid_to_top = np.vstack([np.array((centroid[0], ymax)), centroid])

                        C1 = self.contour_intersect(coords, left_to_centroid)
                        C2 = self.contour_intersect(coords, centroid_to_right)
                        C3 = self.contour_intersect(coords, bottom_to_centroid)
                        C4 = self.contour_intersect(coords, centroid_to_top)

                        count = float(C1) + float(C2) + float(C3) + float(C4)

                        if count >= 3:  # at least on 3 sides, then it's good enough to be considered inside
                            contours_idx_within_ROI.append(other_ROIs_index[jj])

                    # --------------------------------------------------------------------------------------------
                    edges_to_test = lims_to_vec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, patch_size=self.patch_size)

                    # Remove BG in concerned ROIs
                    remove_BG_cond = [ROI_name in remove_bg_contour.lower() for remove_bg_contour in
                                      self.remove_BG_contours]
                    if any(remove_BG_cond):
                        remove_BG = self.remove_BG
                    else:
                        remove_BG = None

                    # Loop over all tiles and see if they are members of the current ROI. Do in // if you have many
                    # tiles, otherwise the overhead cost is not worth it
                    dataset = (coords, self.patch_size, contours_idx_within_ROI, remove_BG, WSI_object, df)

                    if len(edges_to_test) > 5000:

                        with WorkerPool(n_jobs=6, start_method='fork') as pool:
                            pool.set_shared_objects(dataset)
                            results = pool.map(tile_membership, list(edges_to_test), progress_bar=True)

                        membership = np.asarray(results)

                    else:
                        membership = np.full(len(edges_to_test), False)
                        for ei in tqdm(range(len(edges_to_test))):
                            membership[ei] = tile_membership(dataset, edges_to_test[ei, :])

                    coord_x.append(edges_to_test[membership, 0])
                    coord_y.append(edges_to_test[membership, 1])
                    label.append(
                        np.ones(np.sum(membership), dtype=int) * contour_label[0])  # should only be a unique value.

            # Once looped over all ROIs for a given index, export as csv
            coord_x = [item for sublist in coord_x for item in sublist]
            coord_y = [item for sublist in coord_y for item in sublist]
            label = [item for sublist in label for item in sublist]
            df_export = pd.DataFrame({'coords_x': coord_x, 'coords_y': coord_y, self.label_name: label})
            df_export.to_csv(patch_csv_export_filename)
            print('exported at: {}'.format(patch_csv_export_filename))

            # Finally, create the overlay for QA if using contours.
            if contour_files:
                all_possible_labels = np.array(list(set(Full_Contour_Mapping['contour_id'])))  # only unique
                self.create_QA_overlay(df_export, WSI_object, self.patch_size, self.QA_path, ID, all_possible_labels)
                print('QA overlay exported at: {}'.format(
                    os.path.join(self.QA_path, ID + '_patch_' + str(self.patch_size) + '.pdf')))

            print('--------------------------------------------------------------------------------')

        # Once looped over all indexes: export csv that provides the mapping between contour_name and contour_id.
        df_mapping = pd.DataFrame(Full_Contour_Mapping)
        out_mapping = os.path.join(csv_save_dir, 'mapping.csv')
        df_mapping.to_csv(out_mapping)
        print('Mapping exported at: {}'.format(out_mapping))

        return


if __name__ == "__main__":
    config = toml.load('./config_files/tumour_identification_training_ubuntu.ini')
    preprocess = PreProcessor(config)
    preprocess.preprocess_WSI()
