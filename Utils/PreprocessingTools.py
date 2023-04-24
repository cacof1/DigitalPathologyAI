import openslide
import os
import cv2
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Utils import OmeroTools
from PIL import Image
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

def roi_to_points(row):
    # The correct type is polygon.

    # Could add more if useful
    if row['Class'] == 'rectangle':
        xmin = str(row['X'][i])
        xmax = str(row['X'][i] + row['Width'][i])
        ymin = str(row['Y'][i])
        ymax = str(row['Y'][i] + row['Height'][i])
        fullstr = xmin + ',' + ymin + ' ' + xmax + ',' + ymin + ' ' + xmax + ',' + ymax + ' ' + xmin + ',' + ymax
        # clockwise!
        row['Points'] = fullstr

    return row

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
    # shared is a tuple: (WSI_object, patch_size, vis, bg_threshold). To work with // processing.
    # patch_background_fraction grabs an image patch of size patch_size from WSI_object at location edge.
    # It then evaluate background fraction defined as the number of pixels where greyscase > threshold.

    patch = np.array(shared[0].read_region(edge, shared[2], tuple(shared[1])).convert("RGB"))
    patch_norm_gray = patch[:, :, 0] * 0.2989 + patch[:, :, 1] * 0.5870 + patch[:, :, 2] * 0.1140
    background_fraction = np.sum(patch_norm_gray > shared[3]) / np.prod(patch_norm_gray.shape)
    
    return background_fraction

def compare_membership(shared, edge):
    # shared is a tuple: (array of coords in maximum resolution, patch size in maximum resolution)
    
    cond = (shared[0][:,0] > edge[0]) & (shared[0][:,0] <= (edge[0]+shared[1][0])) & (shared[0][:,1] > edge[1]) & (shared[0][:,1] <= (edge[1]+shared[1][1]))

    return np.any(cond)

def tile_membership_contour(shared, edge):
    # shared is a tuple: (patch_size, remove_BG, contours_idx_within_ROI, df, coords).
    # This allows usage with MPIRE for multiprocessing, which provides a modest speedup. Unpack:
    # patch_size = shared[0]
    # remove_BG = shared[1]
    # contours_idx_within_ROI = shared[2]
    # df = shared[3]
    # coords = shared[4]
    # Start by assuming that the patch is within the contour, and remove it if it does not meet a set of conditions.
    
    # First: is the patch within the ROI? Test with cv2 for pre-defined contour,
    # or if no contour then do nothing and use the patch.
    patch_outside_ROI = cv2.pointPolygonTest(shared[4],
                                             (edge[0] + shared[0][0] / 2,
                                              edge[1] + shared[0][1] / 2),
                                             measureDist=False) == -1
    if patch_outside_ROI: return False
    """
    # Second: verify that the valid patch is not within any of the ROIs identified
    # as fully inside the current ROI.
    patch_within_other_ROIs = []
    for ii in range(len(shared[2])):
        cii = shared[2][ii]
        object_in_ROI_coords = split_ROI_points(shared[3]['Points'][cii]).astype(int)
        patch_within_other_ROIs.append(cv2.pointPolygonTest(object_in_ROI_coords,
                                                            (edge[0] + shared[0][0] / 2,
                                                             edge[1] + shared[0][1] / 2),
                                                            measureDist=False) >= 0)
        
    if any(patch_within_other_ROIs): return False
    """
    return True

class Preprocessor:
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

        # For each WSI to be processed, contains slice id
        self.ids = None

        # For each WSI to be processed, contains the index of the dataset to write to in the .npy file
        self.WSI_processing_index = None

        # Maps a contour name from Omero to a list of contour specified in config file

    def Create_Contours_Overlay_QA(self, df_export):

        ## Convert label to numerical value
        le = preprocessing.LabelEncoder()
        numerical_labels      = le.fit_transform(df_export['tissue_type'])
        WSI_object           = openslide.open_slide(df_export['SVS_PATH'].iloc[0])
        vis_level_view       = len(WSI_object.level_dimensions) - 1  # always the lowest res vis level
        N_classes            = len(np.unique(numerical_labels))

        if N_classes <= 10: cmap = plt.get_cmap('Set1', lut=N_classes)           
        elif N_classes > 10 & N_classes <= 20: cmap = plt.get_cmap('tab20', lut=N_classes)        
        else: cmap = plt.get_cmap('Spectral', lut=N_classes)

        cmap.N = N_classes
        cmap.colors = cmap.colors[0:N_classes]

        heatmap, overlay = generate_overlay(WSI_object, numerical_labels + 1, np.array(df_export[["coords_x", "coords_y"]]),
                                            vis_level=vis_level_view, patch_size=self.patch_size, cmap=cmap, alpha=0.4)
                                        
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
        ID = df_export['id_external'].iloc[0]
        img_pth = Path(self.QA_folder, ID + '_patch_' + str(self.patch_size[0]) + '.pdf')
        heatmap_PIL.save(str(img_pth), 'pdf')
        print('QA overlay exported at: {}'.format(Path(self.QA_folder, ID + '_patch_' + str(self.patch_size[0]) + '.pdf')))

    def contours_processing(self, row):

        # Process the contours of the idx^th WSI using the contours from contour_files. If there are no contour files, then each tile is labeled.
        coord_x = []
        coord_y = []

        df         = roi_to_points(row)  # converts ROIs that are not polygons to "Points" for uniform handling.
        print('Found contours, processing.')
        coords = split_ROI_points(df['Points']).astype(int)
        xmin, ymin = np.min(coords, axis=0)
        xmax, ymax = np.max(coords, axis=0)
            
        # To make sure we do not end up with overlapping contours at the end, round xmin, xmax, ymin,
        # ymax to the nearest multiple of self.patch_size.
        xmin = np.floor(xmin / self.patch_size[0]).astype(np.int) * self.patch_size[0]
        ymin = np.floor(ymin / self.patch_size[1]).astype(np.int) * self.patch_size[1]                        
        xmax = np.ceil(xmax / self.patch_size[0]).astype(np.int) * self.patch_size[0]
        ymax = np.ceil(ymax / self.patch_size[1]).astype(np.int) * self.patch_size[1]
        
        # Get the list of all contours that are contained within the current one.
        contours_idx_within_ROI = []
        """
        other_ROIs_index = np.setdiff1d(np.arange(len(df)), i)
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
        """
        edges_to_test = lims_to_vec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,patch_size=self.patch_size)
        remove_BG = None

        # Loop over all tiles and see if they are members of the current ROI. Do in // 
        shared = (self.patch_size, remove_BG, contours_idx_within_ROI, df, coords)
        with WorkerPool(n_jobs=10, start_method='fork') as pool:
            pool.set_shared_objects(shared)
            results = pool.map(tile_membership_contour, list(edges_to_test), progress_bar=False)

        isInROI = np.asarray(results)
        df_export = pd.DataFrame({'coords_x': edges_to_test[isInROI,0], 'coords_y': edges_to_test[isInROI,1], 'tissue_type': row['ROIName']})
        return df_export
    
    # ----------------------------------------------------------------------------------------------------------------
    def getTilesFromAnnotations(self, dataset):

        df = pd.DataFrame()
        for idx, row in dataset.iterrows():  # WSI wise
            print('Processing ROI "{}" ({}/{}) of ID "{}": '.format(row['ROIName'], idx,str(len(dataset)), str(row['id_external'])),end='')                        
            cur_dataset = self.contours_processing(row)
            cur_dataset['id_external'] = row['id_external']
            cur_dataset['SVS_PATH']    = row['SVS_PATH']            
            df = pd.concat([df,cur_dataset], ignore_index=True)
            print('--------------------------------------------------------------------------------')
        df[['coords_x', 'coords_y']] = df[['coords_x', 'coords_y']].astype('int')
        print(df.shape)
        df_final = pd.DataFrame()
        for name, group in df.groupby('id_external'):
            group = group.drop_duplicates(subset=['coords_x','coords_y'], keep='last')
            df_final = pd.concat([df_final, group],ignore_index=True)
            #self.Create_Contours_Overlay_QA(group) For QA
            
        print(df_final.shape)
        return df

    def getAllTiles(self, dataset, background_fraction_threshold=0):

        df = pd.DataFrame()
        for idx, row in dataset.iterrows():
            WSI_object = openslide.open_slide(row['SVS_PATH'])
            print(WSI_object)
            # lowest zoom level edges (assuming processing is done with visibility 0)
            edges_to_test = lims_to_vec(xmin=0, xmax=WSI_object.level_dimensions[0][0], ymin=0,
                                        ymax=WSI_object.level_dimensions[0][1],
                                        patch_size=self.patch_size)

            # remove background in //
            # This is done on the highest zoom level images to accelerate the process.
            downsample_factor    = int(WSI_object.level_downsamples[-1]) # Factor to match high zoom patches coords to low zoom
            high_zoom_patch_size = tuple(np.array(np.array(self.patch_size) / downsample_factor).astype(int))  # Size of patches in high zoom
            high_zoom_vis        = WSI_object.level_count - 1  # visiblity of highest zoom level

            # Compute tile locations for the highest zoom level images
            high_zoom_edges_to_test = lims_to_vec(xmin=0, xmax=WSI_object.level_dimensions[-1][0], ymin=0,
                                                  ymax=WSI_object.level_dimensions[-1][1],
                                                  patch_size=high_zoom_patch_size)
            
            # background threshold is hard coded to 245 (/255) to be highly specific (only remove bg that we are certain)
            shared = (WSI_object, high_zoom_patch_size, high_zoom_vis, 245)
            with WorkerPool(n_jobs=60, start_method='fork') as pool:
                pool.set_shared_objects(shared)
                results = pool.map(patch_background_fraction,
                                   list(downsample_factor * high_zoom_edges_to_test),  
                                   progress_bar=False)
                # (!) Need to multiply high_zoom_edges_to_test by downsample_factor because openslide uses the lowest zoom coordinates for locations.
                
            # Extract the non-background edges and scale them to match the lowest zoom level.
            estimated_low_zoom_non_background_edges = downsample_factor * high_zoom_edges_to_test[np.array(results) < background_fraction_threshold, :]

            shared = (estimated_low_zoom_non_background_edges, self.patch_size)
            with WorkerPool(n_jobs=60, start_method='fork') as pool:
                pool.set_shared_objects(shared)
                results = pool.map(compare_membership, list(edges_to_test), progress_bar = False)

            edges_wo_background = edges_to_test[np.array(results).astype(bool), :]
            # ------------------------------------------------------------------------

            print('Total tiles: {}, total tiles without auto-removed background: {}'.format(len(edges_to_test),len(edges_wo_background)))
            
            cur_dataset = pd.DataFrame({'coords_x': edges_wo_background[:, 0], 'coords_y': edges_wo_background[:, 1]})
            
            cur_dataset['id_external'] = row['id_external']
            cur_dataset['SVS_PATH']    = row['SVS_PATH']            
            df = pd.concat([df, cur_dataset], ignore_index=True)

        print('--------------------------------------------------------------------------------')
        return df
