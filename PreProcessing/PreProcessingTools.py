import openslide
import os
import cv2
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Utils import OmeroTools
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from Visualization.WSI_Viewer import generate_overlay
import copy
from sklearn import preprocessing
sys.path.append('../')
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
    EX, EY  = np.meshgrid(edges_x, edges_y)
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


def patch_background_fraction(shared, edge):
    # shared is a tuple: (WSI_object, patch_size, bg_threshold). To work with // processing.

    # patch_background_fraction grabs an image patch of size patch_size from WSI_object at location edge.
    # It then evaluate background fraction defined as the number of pixels where greyscase > threshold*255.

    patch = np.array(shared[0].read_region(edge, 0, tuple(shared[1])).convert("RGB"))
    img_norm = patch.transpose(2, 0, 1)
    patch_norm_gray = img_norm[:, :, 0] * 0.2989 + img_norm[:, :, 1] * 0.5870 + img_norm[:, :, 2] * 0.1140
    background_fraction = np.sum(patch_norm_gray > shared[2] * 255) / np.prod(patch_norm_gray.shape)

    return background_fraction


def tile_membership_contour(shared, edge):
    # shared is a tuple: (WSI_object, patch_size, remove_BG, contours_idx_within_ROI, df, coords).
    # This allows usage with MPIRE for multiprocessing, which provides a modest speedup. Unpack:
    # WSI_object = shared[0]
    # patch_size = shared[1]
    # remove_BG = shared[2]
    # contours_idx_within_ROI = shared[3]
    # df = shared[4]
    # coords = shared[5]

    # Start by assuming that the patch is within the contour, and remove it if it does not meet a set of conditions.
    
    # First: is the patch within the ROI? Test with cv2 for pre-defined contour,
    # or if no contour then do nothing and use the patch.
    patch_outside_ROI = cv2.pointPolygonTest(shared[5],
                                             (edge[0] + shared[1][0] / 2,
                                              edge[1] + shared[1][1] / 2),
                                             measureDist=False) == -1
    if patch_outside_ROI:
        return False

    # Second: verify that the valid patch is not within any of the ROIs identified
    # as fully inside the current ROI.
    patch_within_other_ROIs = []
    for ii in range(len(shared[3])):
        cii = shared[3][ii]
        object_in_ROI_coords = split_ROI_points(shared[4]['Points'][cii]).astype(int)
        patch_within_other_ROIs.append(cv2.pointPolygonTest(object_in_ROI_coords,
                                                            (edge[0] + shared[1][0] / 2,
                                                             edge[1] + shared[1][1] / 2),
                                                            measureDist=False) >= 0)
        
    if any(patch_within_other_ROIs):
        return False
    return True

class PreProcessor:

    def __init__(self, config):

        self.config = config

        self.patch_size = config['BASEMODEL']['Patch_Size']
        
        # Create some paths that are always the same defined with respect to the data folder.
        self.patches_folder = os.path.join(self.config['DATA']['SVS_Folder'], 'patches')
        os.makedirs(self.patches_folder, exist_ok=True)
        self.QA_folder = os.path.join(self.patches_folder, 'QA')
        os.makedirs(self.QA_folder, exist_ok=True)
        self.contours_folder = Path(self.patches_folder, 'contours')
        os.makedirs(self.contours_folder, exist_ok=True)

                                                                            
        # Tag the session
        e = datetime.datetime.now()
        self.config['INTERNAL'] = dict()
        self.config['INTERNAL']['timestamp'] = "{}_{}_{}, {}h{}m{}s.".format(e.day, e.month, e.year, e.hour,
                                                                             e.minute, e.second)

        # For each WSI to be processed, contains slice id
        self.ids = None

        # For each WSI to be processed, contains the index of the dataset to write to in the .npy file
        self.WSI_processing_index = None

        # Maps a contour name from Omero to a list of contour specified in config file
        self.preprocessing_mapping = {}

    def Create_Contours_Overlay_QA(self, row, df_export):

        ## Convert label to numerical value
        le = preprocessing.LabelEncoder()
        numerical_labels      = le.fit_transform(df_export['tissue_type'])


        WSI_object           = openslide.open_slide(row['SVS_PATH'])
        vis_level_view       = len(WSI_object.level_dimensions) - 1  # always the lowest res vis level
        N_classes            = len(np.unique(numerical_labels))

        if N_classes <= 10: cmap = plt.get_cmap('Set1', lut=N_classes)           
        elif N_classes > 10 & N_classes <= 20: cmap = plt.get_cmap('tab20', lut=N_classes)        
        else: cmap = plt.get_cmap('Spectral', lut=N_classes)
            

        cmap.N = N_classes
        cmap.colors = cmap.colors[0:N_classes]

        heatmap, overlay = generate_overlay(WSI_object, numerical_labels + 1, np.array(df_export[["coords_x", "coords_y"]]),
                                            vis_level=vis_level_view,
                                            patch_size=self.patch_size, cmap=cmap, alpha=0.4)

        # Draw the contours for each label
        heatmap = np.array(heatmap)

        overlay = overlay.astype(np.int64)
        indexes_to_plot = np.unique(overlay[overlay > 0])
        for index_to_plot in indexes_to_plot:
            im1 = 255 * (overlay == index_to_plot).astype(np.uint8)
            contours, hierarchy = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            col = 255 * cmap.colors[index_to_plot - 1]  # because we said label = numerical_labels + 1
            cv2.drawContours(heatmap, contours, -1, col, 3)

        heatmap_PIL = Image.fromarray(heatmap)

        # Export image to QA_path to evaluate the quality of the pre-processing.
        ID = row['id_external']
        img_pth = Path(self.QA_folder, ID + '_patch_' + str(self.patch_size[0]) + '.pdf')
        heatmap_PIL.save(str(img_pth), 'pdf')

        print('QA overlay exported at: {}'.format(Path(self.QA_folder, ID + '_patch_' + str(self.patch_size[0]) + '.pdf')))


    def organise_contours(self,dataset):

        # Creates .csv files from contours, as specified in the configuration file.
        contour_files = []

        # Create contours first. This will download all contours.
        OmeroTools.download_omero_ROIs(self.config, dataset, download_path=self.contours_folder)
        print('--------------------------------------------------------------------------------')

        # Then, load the local contours (same if contour_type is local or omero)
        for ID in dataset['id_external']: contour_files.append(str(Path(self.contours_folder, ID + '_roi_measurements.csv')))

        # The first thing we want to do is assign labels (numbers) to the names of contours. To do so, we must loop over
        # the existing datasets and find all the unique contour types. Also, if using a subset of all contours through
        # "used_contours", make sure you only preserve the relevant contours.

        contour_names = []
        for contour_file in contour_files:
            df = pd.read_csv(contour_file)

            for name in df.Text.unique():

                if all([excluded_contour.lower() not in name.lower() for excluded_contour in self.config['CONTOURS']['Remove_Contours']]):

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
            count_catch_contour = 0

            for ctr_nm in list(unique_contour_names):
                ctr_nm = ctr_nm.replace(' ', '').lower()
                loc = [ctr_nm == key for key in dict_contours_to_map.keys()]
                loc_star = [(key.replace('*', '') in ctr_nm) & ('*' in key) for key in dict_contours_to_map.keys()]

                if any(loc):  # if the contour exists in the mapping
                    unique_contour_names_mapped.append(dict_contours_to_map[ctr_nm])

                elif any(loc_star):  # if the contour exists, but it's in the format contour*
                    dict_key = [k for ki, k in enumerate(dict_contours_to_map.keys()) if loc_star[ki]][0]
                    unique_contour_names_mapped.append(dict_contours_to_map[dict_key])

                elif 'remaining' in keys:  # shortcut to assign all other contours to dict_contours_to_map['remaining']
                    unique_contour_names_mapped.append(dict_contours_to_map['remaining'])

                else:  # otherwise, use the contour, do not modify it.
                    unique_contour_names_mapped.append(ctr_nm)
                    count_catch_contour += 1

        else:
            unique_contour_names_mapped = unique_contour_names

        self.preprocessing_mapping = dict(zip(unique_contour_names, unique_contour_names_mapped))

        return contour_files

    def contours_processing(self, row):

        # Process the contours of the idx^th WSI using the contours from contour_files. If there are no contour files, then each tile is labeled.
        coord_x = []
        coord_y = []
        label   = []

        #  load contours if existing, or process entire WSI
        WSI_object = openslide.open_slide(row['SVS_PATH'])

        df = pd.read_csv(row['contour_file'])
        df = roi_to_points(df)  # converts ROIs that are not polygons to "Points" for uniform handling.

        # Loop over each contour and extract patches contained within
        for i in range(len(df)):
            ROI_name = df['Text'][i].lower()
            print('Processing ROI "{}" ({}/{}) of ID "{}": '.format(ROI_name, str(i + 1), str(len(df)), str(row['id_external'])),end='')            
            if ROI_name not in self.preprocessing_mapping.keys():
                print('ROI not within selected contours, skipping.')
            else:
                print('Found contours, processing.')
                coords = split_ROI_points(df['Points'][i]).astype(int)
                xmin, ymin = np.min(coords, axis=0)
                xmax, ymax = np.max(coords, axis=0)
                
                # To make sure we do not end up with overlapping contours at the end, round xmin, xmax, ymin,
                # ymax to the nearest multiple of self.patch_size.
                ps = self.patch_size
                xmin = int(np.floor(xmin / ps[0]) * ps[0])
                ymin = int(np.floor(ymin / ps[1]) * ps[1])
                xmax = int(np.ceil(xmax / ps[0]) * ps[0])
                ymax = int(np.ceil(ymax / ps[1]) * ps[1])
                        
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
                edges_to_test = lims_to_vec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,patch_size=self.patch_size)

                # Remove BG in concerned ROIs
                #remove_BG_cond = [ROI_name.lower() in remove_bg_contour.lower() for remove_bg_contour in self.config['CONTOURS']['Background_Removal']]
                #if any(remove_BG_cond): remove_BG = self.config['CONTOURS']['Background_Thresh']
                #else:remove_BG = None
                remove_BG = None
                # Loop over all tiles and see if they are members of the current ROI. Do in // if you have many tiles, otherwise the overhead cost is not worth it
                shared = (WSI_object, self.patch_size, remove_BG, contours_idx_within_ROI, df, coords)
                
                if len(edges_to_test) > 1500:  # minimal number for // processing, otherwise not worth it.
                    with WorkerPool(n_jobs=10, start_method='fork') as pool:
                        pool.set_shared_objects(shared)
                        results = pool.map(tile_membership_contour, list(edges_to_test), progress_bar=True)

                    isInROI = np.asarray(results)

                else:
                    isInROI = np.full(len(edges_to_test), False)
                    for ei in tqdm(range(len(edges_to_test))): isInROI[ei] = tile_membership_contour(shared, edges_to_test[ei, :])

                coord_x.extend(edges_to_test[isInROI, 0])
                coord_y.extend(edges_to_test[isInROI, 1])
                label.extend(np.full(len(np.where(isInROI)[0]), self.preprocessing_mapping[ROI_name]))

        df_export = pd.DataFrame({'coords_x': coord_x, 'coords_y': coord_y, 'tissue_type': label})
        return df_export
    
    # ----------------------------------------------------------------------------------------------------------------
    def getTilesFromAnnotations(self, dataset):

        # Download and organise contours
        if self.config['CONTOURS']: dataset['contour_file'] = self.organise_contours(dataset)
        df = pd.DataFrame()
        # process the dataset and export to npy.
        for idx, row in dataset.iterrows():  # WSI wise
            cur_dataset = self.contours_processing(row)
            cur_dataset['SVS_ID'] = row['id_external']
            self.Create_Contours_Overlay_QA(row, cur_dataset)
            df = df.append(cur_dataset, ignore_index=True)
            print('--------------------------------------------------------------------------------')

        df[['coords_x', 'coords_y']] = df[['coords_x', 'coords_y']].astype('int')
        return df

    def getAllTiles(self, dataset):

        df = pd.DataFrame()
        for idx, row in dataset.iterrows():

            WSI_object = openslide.open_slide(row['SVS_PATH'])
            edges_to_test = lims_to_vec(xmin=0, xmax=WSI_object.level_dimensions[0][0], ymin=0,
                                        ymax=WSI_object.level_dimensions[0][1],
                                        patch_size=self.patch_size)
            cur_dataset = pd.DataFrame({'coords_x': edges_to_test[:, 0], 'coords_y': edges_to_test[:, 1]})
            cur_dataset['SVS_ID'] = row['id_external']
            df = pd.concat([df, cur_dataset], ignore_index=True)

        print('--------------------------------------------------------------------------------')
        return df


