import glob
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils.OmeroTools
from wsi_core.WholeSlideImage import WholeSlideImage
from PIL import Image
from tqdm import tqdm
from joblib import delayed, Parallel
import multiprocessing as mp


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


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


def split_ROI_points(coords_string):
    coords = np.array([[float(coord_string.split(',')[0]), float(coord_string.split(',')[1])] for coord_string in
                       coords_string.split(' ')])
    return coords


def plot_contour(xmin, xmax, ymin, ymax, colour='k'):
    plt.plot([xmin, xmax], [ymin, ymin], '-' + colour)
    plt.plot([xmin, xmax], [ymax, ymax], '-' + colour)
    plt.plot([xmin, xmin], [ymin, ymax], '-' + colour)
    plt.plot([xmax, xmax], [ymin, ymax], '-' + colour)


def create_QA_overlay(df_export, WSI_object, patch_size, QA_path, ID, all_possible_labels):

    vis_level_view = 3
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
    legend_coords = np.array([np.arange(0, N_classes*patch_size, patch_size), np.zeros(N_classes)]).T
    legend_label = all_possible_labels
    all_labels = np.concatenate((legend_label, df_export['label'].values + 1), axis=0)
    all_coords = np.concatenate((legend_coords ,np.array(df_export[["coords_x", "coords_y"]])), axis=0)

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
    pi = np.array(heatmap)
    indexes_to_plot = np.unique(overlay[overlay > 0])
    for ii in range(len(indexes_to_plot)):
        im1 = 255 * (overlay == indexes_to_plot[ii]).astype(np.uint8)
        contours, hierarchy = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        color_label = np.argwhere(indexes_to_plot[ii] == all_possible_labels+1)[0][0]
        print(color_label)
        cv2.drawContours(pi, contours, -1, 255 * cmap.colors[color_label], 3)

    piPIL = Image.fromarray(pi)
    piPIL.show()

    # Export image to QA_path to evaluate the quality of the pre-processing.
    os.makedirs(QA_path, exist_ok=True)
    img_pth = os.path.join(QA_path, ID + '_patch_' + str(patch_size) + '.pdf')
    piPIL.save(img_pth, 'pdf')


def tile_membership(edge, coords, patch_size, contours_idx_within_ROI, remove_outliers, WSI_object, df):

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

    # Another condition: verify if the remove outlier condition is turned on, and apply.
    # This can be a bit slow as we need to load each patch and assess its colour.
    if remove_outliers:

        patch = np.array(
            WSI_object.wsi.read_region(edge, 0, (patch_size, patch_size)).convert("RGB"))
        average_colour = np.mean(patch.reshape(patch_size ** 2, 3), axis=0)
        patch_too_white = np.all(average_colour > remove_outliers * np.array([255.0, 255.0, 255.0]))

        if patch_too_white:
            return False
        patch_too_dark = np.all(average_colour < remove_outliers * np.array([1.0, 1.0, 1.0]))

        if patch_too_dark:
            return False

        # for debugging/validation purposes: uncomment to plot all patches on top of contours.
        # plot_contour(edge[0], edge[0] + patch_size, edge[1], edge[1] + patch_size)

    return True


def preprocess_WSI(vis=0, patch_size=256, patch_path=None, svs_path=None, ids=None,
                   contour_path=None, contour_type='local', QA_path=None,
                   specific_contours=False, omero_login=None, remove_outliers=None):
    # LIST OF INPUTS
    # -------------------------------- Required -------------------------------------
    # vis                : visibility level, leave at 0 for improved performance.
    # patch_size         : scalar value (256, 512, etc)
    # patch_path         : folder where .csv files of pre-processed WSI will be saved.
    # patch_path         : if using contours, folder where images of the processed WSI will be saved for QA.
    # svs_path           : parent path of all svs files. Some files can be located within subfolders of svs_path.
    # ids                : list of ids to process. If empty, will do all slices found recursively in "svs_path".
    # -------------------------------- Optional -------------------------------------
    # contour_path       : string where contours are located or will be located, depending on contour_type below.
    #                      If empty, no contours will be used and the .csv will be generated using all tiles.
    # contour_type       : options: 'local', 'omero'. If omero, then the 'omero_login' dict must be provided.
    # specific_contours  : a list of contours that will be used; the rest are not used. If empty, uses all contours.
    # omero_login        : dict with fields required to run utils.OmeroTools.download_omero_ROIs.
    #                      Used if contour_type=='omero'.
    # remove_outliers    : if set to a scalar value, this will remove, for each contour that you consider, all patches
    #                      whose average colour is remove_outliers *[0, 0, 0] or remove_outliers * [255, 255, 255];
    #                      in other words, patches that are almost all white or all black are removed from the analysis,
    #                      as they provide no relevant information.

    # Create patch exportation folder and score file names
    csv_save_dir = os.path.join(patch_path, 'csv_vis' + str(vis) + '_patch' + str(patch_size))
    os.makedirs(csv_save_dir, exist_ok=True)

    if contour_path != '':
        contour_files = []

        # Create contours first. This will download all contours.
        if contour_type == 'omero':
            utils.OmeroTools.download_omero_ROIs(download_path=contour_path, **omero_login)
            print('--------------------------------------------------------------------------------')

        # Then, load the local contours (same if contour_type is local or omero)
        for ID in ids:
            contour_files.append(os.path.join(contour_path, ID + '.svs [0]_roi_measurements.csv'))

    else:
        contour_files = []

    # The first thing we want to do is assign labels (numbers) to the names of contours. To do so, we must loop over
    # the existing datasets and find all the unique contour types. Also, if using a subset of all contours through
    # "used_contours", make sure you only preserve the relevant contours.

    contour_names = []
    for contour_file in contour_files:
        df = pd.read_csv(os.path.join(contour_path, contour_file))

        for name in df.Text.unique():

            if specific_contours:
                if name in specific_contours:
                    contour_names.append(name)
            else:
                contour_names.append(name)

    contour_mapping = {'contour_name': list(set(contour_names)), 'contour_id': np.arange(len(set(contour_names)))}

    # Loop over each file and extract relevant indexes.
    for idx in range(len(ids)):

        # Preallocate some arrays
        coord_x = []
        coord_y = []
        label = []
        ID = ids[idx]
        patch_csv_export_filename = os.path.join(csv_save_dir, ID + '.csv')

        #  load contours if existing, or process entire WSI
        search_WSI_query = os.path.join(svs_path, '**', ID + '.svs')
        svs_filename = glob.glob(search_WSI_query, recursive=True)[0]  # if file is hidden recursively
        WSI_object = WholeSlideImage(svs_filename)
        if contour_files:
            df = pd.read_csv(contour_files[idx])
        else:
            xmax, ymax = WSI_object.level_dim[0]
            xmin, ymin = 0, 0
            df = [1]  # dummy placeholder as we have a single contour equal to the entire image.

        # Loop over each contour and extract patches contained within
        for i in range(len(df)):

            ROI_name = df['Text'][i]

            print('Processing ROI "{}" ({}/{}) of ID "{}": '.format(ROI_name, str(i + 1), str(len(df)),str(ID)), end='')

            if ROI_name not in contour_mapping['contour_name']:

                print('ROI not within selected contours, skipping.')

            else:

                print('Found contours, processing.')

                if isinstance(df, pd.DataFrame):
                    coords = split_ROI_points(df['Points'][i]).astype(int)
                    contour_label = [l for s, l in zip(contour_mapping['contour_name'], contour_mapping['contour_id'])
                                     if
                                     ROI_name in s]

                    xmin, ymin = np.min(coords, axis=0)
                    xmax, ymax = np.max(coords, axis=0)

                    # To make sure we do not end up with overlapping contours at the end, round xmin, xmax, ymin,
                    # ymax to the nearest multiple of patch_size.
                    xmin = int(np.floor(xmin / patch_size) * patch_size)
                    ymin = int(np.floor(ymin / patch_size) * patch_size)
                    xmax = int(np.ceil(xmax / patch_size) * patch_size)
                    ymax = int(np.ceil(ymax / patch_size) * patch_size)

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
                # Create an array containing all relevant tile edges
                edges_x = np.arange(xmin, xmax, patch_size)
                edges_y = np.arange(ymin, ymax, patch_size)
                EX, EY = np.meshgrid(edges_x, edges_y)
                edges_to_test = np.column_stack((EX.flatten(), EY.flatten()))

                # Loop over all tiles and see if they are members of the current ROI
                membership = np.full(len(edges_to_test), False)
                for ei in tqdm(range(len(edges_to_test))):
                    membership[ei] = tile_membership(edges_to_test[ei, :], coords, patch_size, contours_idx_within_ROI,
                                                     remove_outliers, WSI_object, df)

                coord_x.append(edges_to_test[membership, 0])
                coord_y.append(edges_to_test[membership, 1])
                label.append(np.ones(np.sum(membership), dtype=int)*contour_label[0])  # should only be a single value.


        # Once looped over all ROIs for a given index, export as csv
        coord_x = [item for sublist in coord_x for item in sublist]
        coord_y = [item for sublist in coord_y for item in sublist]
        label = [item for sublist in label for item in sublist]
        df_export = pd.DataFrame({'coords_x': coord_x, 'coords_y': coord_y, 'label': label})
        df_export.to_csv(patch_csv_export_filename)
        print('exported at: {}'.format(patch_csv_export_filename))

        # Finally, create the overlay for QA if using contours.
        if contour_files:
            all_possible_labels = contour_mapping['contour_id']
            create_QA_overlay(df_export, WSI_object, patch_size, QA_path, ID, all_possible_labels)
            print('QA overlay exported at: {}'.format(os.path.join(QA_path, ID + '_patch_' + str(patch_size) + '.pdf')))

    # Once looped over all indexes: export csv that provides the mapping between contour_name and contour_id.
    df_mapping = pd.DataFrame(contour_mapping)
    out_mapping = os.path.join(csv_save_dir, 'mapping.csv')
    df_mapping.to_csv(out_mapping)
    print('--------------------------------------------')
    print('Mapping exported at: {}'.format(out_mapping))

    return