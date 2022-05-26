import openslide
import glob
import os
import cv2
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Utils import OmeroTools, npyExportTools
from PIL import Image
from tqdm import tqdm
import torch
from WSI_Viewer import generate_overlay
from Dataloader.Dataloader import gather_WSI_npy_indexes
import copy

sys.path.append('../')
import toml
from QA.Normalization.Colour import ColourNorm
from mpire import WorkerPool
import datetime
from sys import platform


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


def lims_to_vec(xmin=0, xmax=0, ymin=0, ymax=0, patch_size=[0, 0]):
    # Create an array containing all relevant tile edges
    edges_x = np.arange(xmin, xmax, patch_size[0])
    edges_y = np.arange(ymin, ymax, patch_size[1])
    EX, EY = np.meshgrid(edges_x, edges_y)
    edges_to_test = np.column_stack((EX.flatten(), EY.flatten()))
    return edges_to_test


def contour_intersect(cnt_ref, cnt_query):
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


def patch_background_fraction(dataset, edge):
    # dataset is a tuple: (WSI_object, patch_size, colour_normaliser, bg_threshold). To work with // processing.

    # patch_background_fraction grabs an image patch of size patch_size from WSI_object at location edge.
    # It then applies colour normalisation with using the colour_normaliser class and evaluate background fraction,
    # defined as the number of pixels whose grayscale intensity is > threshold*255.

    patch = np.array(dataset[0].read_region(edge, 0, tuple(dataset[1])).convert("RGB"))

    if dataset[2]:
        img_norm, _, _ = dataset[2].normalize(torch.tensor(patch).permute(2, 0, 1), stains=False)
        img_norm = img_norm.numpy()
    else:
        img_norm = patch.transpose(2, 0, 1)

    patch_norm_gray = img_norm[:, :, 0] * 0.2989 + img_norm[:, :, 1] * 0.5870 + img_norm[:, :, 2] * 0.1140
    background_fraction = np.sum(patch_norm_gray > dataset[3] * 255) / np.prod(patch_norm_gray.shape)

    return background_fraction


def tile_membership_contour(dataset, edge):
    # dataset is a tuple: (WSI_object, patch_size, colour_normaliser, remove_BG, contours_idx_within_ROI, df, coords).
    # This allows usage with MPIRE for multiprocessing, which provides a modest speedup. Unpack:
    # WSI_object = dataset[0]
    # patch_size = dataset[1]
    # colour_normaliser = dataset[2]
    # remove_BG = dataset[3]
    # contours_idx_within_ROI = dataset[4]
    # df = dataset[5]
    # coords = dataset[6]

    # Start by assuming that the patch is within the contour, and remove it if it does not meet
    # a set of conditions.
    if isinstance(dataset[5], pd.DataFrame):

        # First: is the patch within the ROI? Test with cv2 for pre-defined contour,
        # or if no contour then do nothing and use the patch.
        patch_outside_ROI = cv2.pointPolygonTest(dataset[6],
                                                 (edge[0] + dataset[1][0] / 2,
                                                  edge[1] + dataset[1][1] / 2),
                                                 measureDist=False) == -1
        if patch_outside_ROI:
            return False

        # Second: verify that the valid patch is not within any of the ROIs identified
        # as fully inside the current ROI.
        patch_within_other_ROIs = []
        for ii in range(len(dataset[4])):
            cii = dataset[4][ii]
            object_in_ROI_coords = split_ROI_points(dataset[5]['Points'][cii]).astype(int)
            patch_within_other_ROIs.append(cv2.pointPolygonTest(object_in_ROI_coords,
                                                                (edge[0] + dataset[1][0] / 2,
                                                                 edge[1] + dataset[1][1] / 2),
                                                                measureDist=False) >= 0)

        if any(patch_within_other_ROIs):
            return False

    # Third condition: verify if the remove background condition is turned on, and apply.
    # This is the code's main bottleneck as we need to load each patch, colour-normalise it, and assess its colour.
    if dataset[3]:

        background_fraction = patch_background_fraction(dataset[0:4], edge)

        if background_fraction > 0.5:
            return False

        # for debugging/validation purposes: uncomment to plot all patches on top of contours.
        # plot_contour(edge[0], edge[0] + patch_size, edge[1], edge[1] + patch_size)

    return True


class PreProcessor:

    def __init__(self, config):

        self.config = copy.deepcopy(config)

        # Robustness to various forms of Vis
        self.vis = copy.copy(config['DATA']['Vis'][0])
        if len(config['DATA']['Vis']) > 1:
            print('Unsupported number of visibility levels, using the first one: {}'.format(self.vis))

        # Robustness to various forms of Patch_Size
        self.patch_size = copy.copy(config['DATA']['Patch_Size'][0])
        if len(config['DATA']['Patch_Size']) > 1:
            print('Unsupported number of patch sizes, using the first one: {}'.format(self.patch_size))


        if 'Colour_Norm_File' in config['NORMALIZATION']:
            self.colour_normaliser = ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File'])
        else:
            self.colour_normaliser = None

        # Create some paths that are always the same defined with respect to the data folder.
        self.patches_folder = os.path.join(self.config['DATA']['SVS_Folder'], 'patches')
        os.makedirs(self.patches_folder, exist_ok=True)
        self.QA_folder = os.path.join(self.patches_folder, 'QA')
        os.makedirs(self.QA_folder, exist_ok=True)
        self.contours_folder = os.path.join(self.patches_folder, 'contours')
        os.makedirs(self.contours_folder, exist_ok=True)

        # Format OMERO login as a dict for easier access.
        if 'OMERO' in config:
            self.omero_login = {'host': config['OMERO']['Host'], 'user': config['OMERO']['User'],
                                'pw': config['OMERO']['Pw'],
                                'target_group': config['OMERO']['Target_Group']}
        else:
            self.omero_login = None

        # Tag the session
        e = datetime.datetime.now()
        self.config['INTERNAL'] = dict()
        self.config['INTERNAL']['timestamp'] = "{}_{}_{}, {}h{}m{}s.".format(e.day, e.month, e.year, e.hour,
                                                                             e.minute, e.second)

        # For each WSI to be processed, contains slice id
        self.ids = None

        # For each WSI to be processed, contains the index of the dataset to write to in the .npy file
        self.WSI_processing_index = None

    # basic functions designed for AnnotationsToNPY
    # ----------------------------------------------------------------------------------------------------------------
    def openslide_read_WSI(self, id):
        search_WSI_query = os.path.join(self.config['DATA']['SVS_Folder'], '**', id + '.svs')
        svs_filename = glob.glob(search_WSI_query, recursive=True)[0]  # if file is hidden recursively
        WSI_object = openslide.open_slide(svs_filename)
        return WSI_object

    def Create_Contours_Overlay_QA(self, idx, df_export):

        patch_size = self.patch_size
        all_possible_labels = np.array(list(set(self.config['PREPROCESSING_MAPPING']['contour_id'].values)))
        WSI_object = self.openslide_read_WSI(self.ids[idx])
        vis_level_view = len(WSI_object.level_dimensions) - 1  # always the lowest res vis level
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
        legend_coords = np.array([np.arange(0, N_classes * patch_size[0], patch_size[0]), np.zeros(N_classes)]).T
        legend_label = all_possible_labels
        all_labels = np.concatenate((legend_label, df_export[self.config['DATA']['Label_Name']].values + 1), axis=0)
        all_coords = np.concatenate((legend_coords, np.array(df_export[["coords_x", "coords_y"]])), axis=0)

        # Broken for now - will be fixed in the next update. This will just not display QA maps.
        heatmap, overlay = generate_overlay(WSI_object, all_labels, all_coords, vis_level=vis_level_view,
                                            patch_size=patch_size, cmap=cmap, alpha=0.4)

        # Draw the contours for each label
        heatmap = np.array(heatmap)
        indexes_to_plot = np.unique(overlay[overlay > 0])
        for ii in range(len(indexes_to_plot)):
            im1 = 255 * (overlay == indexes_to_plot[ii]).astype(np.uint8)
            contours, hierarchy = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            color_label = np.argwhere(indexes_to_plot[ii] == all_possible_labels + 1)

            if len(color_label) > 0:
                col = 255 * cmap.colors[color_label[0][0]]
            else:
                col = [0, 0, 0]  # black contour if you can't find a colour.

            cv2.drawContours(heatmap, contours, -1, col, 3)

        heatmap_PIL = Image.fromarray(heatmap)
        # heatmap_PIL.show()

        # Export image to QA_path to evaluate the quality of the pre-processing.
        ID = self.ids[idx]
        img_pth = os.path.join(self.QA_folder, ID + '_patch_' + str(patch_size[0]) + '.pdf')
        heatmap_PIL.save(img_pth, 'pdf')

        print('QA overlay exported at: {}'.format(
            os.path.join(self.QA_folder, ID + '_patch_' + str(self.patch_size[0]) + '.pdf')))

    def organise_contours(self):

        # Creates .csv files from contours, as specified in the configuration file.
        contour_files = []

        # Create contours first. This will download all contours.
        OmeroTools.download_omero_ROIs(download_path=self.contours_folder, ids=self.ids,
                                       **self.omero_login)
        print('--------------------------------------------------------------------------------')

        # Then, load the local contours (same if contour_type is local or omero)
        for ID in self.ids:
            contour_files.append(
                os.path.join(self.contours_folder, ID + '.svs [0]_roi_measurements.csv'))

        # The first thing we want to do is assign labels (numbers) to the names of contours. To do so, we must loop over
        # the existing datasets and find all the unique contour types. Also, if using a subset of all contours through
        # "used_contours", make sure you only preserve the relevant contours.

        contour_names = []
        for contour_file in contour_files:
            df = pd.read_csv(os.path.join(self.contours_folder, contour_file))

            for name in df.Text.unique():

                if all([excluded_contour.lower() not in name.lower() for excluded_contour in
                        self.config['CONTOURS']['Remove_Contours']]):

                    if self.config['CONTOURS']['Specific_Contours']:
                        if name in self.config['CONTOURS']['Specific_Contours']:
                            contour_names.append(name.lower())
                    else:
                        contour_names.append(name.lower())

        unique_contour_names = list(set(contour_names))

        print('CONTOURS: List of all contours:')
        print(unique_contour_names)
        print('--------------------------------------------------------------------------------')

        # contour_names gives the name of all existing contours. In some cases, you might want to group some contours
        # together, for instance all artifacts together, all fat [in] or [out] together, etc. We create a mapping
        # from contour_names to contour_names_mapped to do so. If contour_mapping = '', skip this step and just process
        # normally each contour.
        if isinstance(self.config['CONTOURS']['Contour_Mapping'], list):

            # Do some pre-processing to locate keys: remove spaces and use lowercase.
            keys = [ctr.split(':')[0].replace(' ', '').lower() for ctr in self.config['CONTOURS']['Contour_Mapping']]
            values = [ctr.split(':')[1] for ctr in self.config['CONTOURS']['Contour_Mapping']]
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

        # Create pd dataframe that provides the mapping between contour_name and contour_id, add it to header.
        self.config['PREPROCESSING_MAPPING'] = pd.DataFrame(Full_Contour_Mapping)
        # out_mapping = os.path.join(self.patches_folder, 'mapping.csv')
        # df_mapping.to_csv(out_mapping)
        # print('Mapping exported at: {}'.format(out_mapping))

        return contour_files

    def contours_processing(self, contour_files, idx):

        # Process the contours of the idx^th WSI using the contours from contour_files. If there are no contour files,
        # then each tile is labeled.

        # Preallocate some arrays
        coord_x = []
        coord_y = []
        label = []
        ID = self.ids[idx]

        #  load contours if existing, or process entire WSI
        WSI_object = self.openslide_read_WSI(ID)

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

                xmax, ymax = WSI_object.level_dimensions[0]
                edges_to_test = lims_to_vec(xmin=0, xmax=xmax, ymin=0, ymax=ymax,
                                            patch_size=self.patch_size)
                coord_x.append(edges_to_test[:, 0])
                coord_y.append(edges_to_test[:, 1])
                label.append(np.ones(len(edges_to_test), dtype=int) * -1)  # should only be a single value.

            elif ROI_name not in self.config['PREPROCESSING_MAPPING']['contour_name'].values:

                print('ROI not within selected contours, skipping.')

            else:

                print('Found contours, processing.')

                if isinstance(df, pd.DataFrame):
                    coords = split_ROI_points(df['Points'][i]).astype(int)
                    contour_label = [l for s, l in
                                     zip(self.config['PREPROCESSING_MAPPING']['contour_name'].values,
                                         self.config['PREPROCESSING_MAPPING']['contour_id'].values)
                                     if ROI_name in s]

                    xmin, ymin = np.min(coords, axis=0)
                    xmax, ymax = np.max(coords, axis=0)

                    # To make sure we do not end up with overlapping contours at the end, round xmin, xmax, ymin,
                    # ymax to the nearest multiple of patch_size.
                    xmin = int(
                        np.floor(xmin / self.patch_size[0]) * self.patch_size[0])
                    ymin = int(
                        np.floor(ymin / self.patch_size[1]) * self.patch_size[1])
                    xmax = int(
                        np.ceil(xmax / self.patch_size[0]) * self.patch_size[0])
                    ymax = int(
                        np.ceil(ymax / self.patch_size[1]) * self.patch_size[1])

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

                    C1 = contour_intersect(coords, left_to_centroid)
                    C2 = contour_intersect(coords, centroid_to_right)
                    C3 = contour_intersect(coords, bottom_to_centroid)
                    C4 = contour_intersect(coords, centroid_to_top)

                    count = float(C1) + float(C2) + float(C3) + float(C4)

                    if count >= 3:  # at least on 3 sides, then it's good enough to be considered inside
                        contours_idx_within_ROI.append(other_ROIs_index[jj])

                # --------------------------------------------------------------------------------------------
                edges_to_test = lims_to_vec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                            patch_size=self.patch_size)

                # Remove BG in concerned ROIs
                remove_BG_cond = [ROI_name.lower() in remove_bg_contour.lower() for remove_bg_contour in
                                  self.config['CONTOURS']['Background_Removal']]
                if any(remove_BG_cond):
                    remove_BG = self.config['CONTOURS']['Background_Thresh']
                else:
                    remove_BG = None

                # Loop over all tiles and see if they are members of the current ROI. Do in // if you have many
                # tiles, otherwise the overhead cost is not worth it
                dataset = (WSI_object, self.patch_size, self.colour_normaliser,
                           remove_BG, contours_idx_within_ROI, df, coords)

                if len(edges_to_test) > 1500 and (
                        platform != "darwin"):  # minimal number for // processing, otherwise not worth it.

                    with WorkerPool(n_jobs=10, start_method='fork') as pool:
                        pool.set_shared_objects(dataset)
                        results = pool.map(tile_membership_contour, list(edges_to_test), progress_bar=True)

                    membership = np.asarray(results)

                else:
                    membership = np.full(len(edges_to_test), False)
                    for ei in tqdm(range(len(edges_to_test)), desc="Background estimation of ID #{}...".format(ID)):
                        membership[ei] = tile_membership_contour(dataset, edges_to_test[ei, :])

                coord_x.append(edges_to_test[membership, 0])
                coord_y.append(edges_to_test[membership, 1])
                label.append(
                    np.ones(np.sum(membership), dtype=int) * contour_label[0])  # should only be a unique value.

        # Once looped over all ROIs for a given index, export as csv
        coord_x = [item for sublist in coord_x for item in sublist]
        coord_y = [item for sublist in coord_y for item in sublist]
        label = [item for sublist in label for item in sublist]
        df_export = pd.DataFrame(
            {'coords_x': coord_x, 'coords_y': coord_y, self.config['DATA']['Label_Name']: label})

        cur_dataset = {'header': self.config, 'dataframe': df_export}

        return cur_dataset

    def labels_processing(self, idx):

        ID = self.ids[idx]

        # 1. Load WSI
        WSI_object = self.openslide_read_WSI(ID)

        # 2. Get label
        WSI_label = -1  # TODO: add code that gets label from OMERO. For now, set value to -1 as a dummy.
        # Will need to use self.config['DATA']['Label_Name'] to gather the relevant key/value from Omero WSIs.

        # 3. Identify non-background patches
        edges_to_test = lims_to_vec(xmin=0, xmax=WSI_object.level_dimensions[0][0], ymin=0,
                                    ymax=WSI_object.level_dimensions[0][1],
                                    patch_size=self.patch_size)
        dataset = (WSI_object, self.patch_size,
                   self.colour_normaliser, 0.86)  # for // processing
        # TODO: for now, background threshold value is hard coded to 0.86, following literature values

        if platform != "darwin":  # fail safe - parallel computing not working on M1 macs for now
            with WorkerPool(n_jobs=10, start_method='fork') as pool:
                pool.set_shared_objects(dataset)
                background_fractions = pool.map(patch_background_fraction, list(edges_to_test), progress_bar=True)
            background_fractions = np.asarray(background_fractions)

        else:
            background_fractions = np.zeros(len(edges_to_test))
            for ii in tqdm(range(len(edges_to_test))):
                background_fractions[ii] = patch_background_fraction(dataset, edges_to_test[ii, :])

        valid_patches = background_fractions < 0.5

        # 4. Create dataframe
        coord_x = edges_to_test[valid_patches, 0]
        coord_y = edges_to_test[valid_patches, 1]
        label = np.ones(np.sum(valid_patches), dtype=int) * WSI_label
        df_out = pd.DataFrame({'coords_x': coord_x, 'coords_y': coord_y, self.config['DATA']['Label_Name']: label})

        cur_dataset = {'header': self.config, 'dataframe': df_out}

        return cur_dataset

    def export_to_npy(self, idx, cur_dataset):

        n = self.WSI_processing_index[idx]
        cid = self.ids[idx]
        patch_npy_export_filename = os.path.join(self.patches_folder, cid + '.npy')

        if os.path.exists(patch_npy_export_filename):  # either append or replace

            datasets = list(np.load(patch_npy_export_filename, allow_pickle=True))
            N = len(datasets)

            if n == N:  # append to file
                datasets.append(cur_dataset)
                np.save(patch_npy_export_filename, datasets)
                print('WSI {}.npy: appended current dataset to existing file.'.format(cid))

            elif (n >= 0) and (n < N):
                datasets[n]['dataframe'] = pd.merge(datasets[n]['dataframe'], cur_dataset['dataframe'])
                print('WSI {}.npy: merged current dataset with previous dataset of existing file.'.format(cid))
                # TODO: test if merge works. Has not been validated yet.

        else:  # create new file
            np.save(patch_npy_export_filename, [cur_dataset])
            print('WSI {}.npy: new file created and added current dataset.'.format(cid))

        # Also output an excel sheet of the svs. For debugging purposes.
        npyExportTools.decode_npy(patch_npy_export_filename)

    # ----------------------------------------------------------------------------------------------------------------
    def AnnotationsToNPY(self, ids, overwrite=True):

        self.ids = ids

        # --------------
        # 1. Download and organise contours, if you are currently working with contours.
        if self.config['CONTOURS']:
            contour_files = self.organise_contours()
        else:
            contour_files = []
        # --------------

        # --------------
        # 2. Gather the index from the .npy file that will be used to store the results of the current pre-processing.
        self.WSI_processing_index, processing_flag = gather_WSI_npy_indexes(self.config, self.ids, overwrite, verbose=True)
        # --------------

        # --------------
        # 3 and 4, performed WSI-wise: process the dataset and export to npy.
        for idx in range(len(self.ids)):  # WSI wise

            if processing_flag[idx]:

                # 4. Process the dataset if it has contours or not
                if self.config['CONTOURS']:
                    cur_dataset = self.contours_processing(contour_files, idx)
                    if contour_files:
                        self.Create_Contours_Overlay_QA(idx, cur_dataset['dataframe'])
                else:
                    cur_dataset = self.labels_processing(idx)

                # 5. Export to npy.
                self.export_to_npy(idx, cur_dataset)
            # --------------

            print('--------------------------------------------------------------------------------')

        return


if __name__ == "__main__":
    config = toml.load('./Configs/example_config.ini')
    preprocessor = PreProcessor(config)


