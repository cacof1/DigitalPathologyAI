
import nrrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import openslide
from matplotlib import patches
import cv2

def get_bbox_from_mask(mask):
    pos = np.where(mask==255)
    if pos[0].shape[0] == 0:
        return np.zeros((0,4))
    else:
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

diagnosis = 'synovial_sarcoma'
#SFT_high
#desmoid_fibromatosis
#superficial_fibromatosis
#df = pd.read_csv('/home/dgs1/data/DigitalPathologyAI/MitoticDetection/omero/{}_MF_ROIs.csv'.format(diagnosis))
#df = pd.read_csv('/home/dgs1/data/DigitalPathologyAI/MitoticDetection/all_tiles_0208.csv')
source = 'AI'
df = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/DetectionResults/classification_coords_synovial_sarcoma0.csv')
df = df.astype({'SVS_ID':'str'})
df = df[df['prob_yes']>0.5]
df = df[df['scores']>0.8].reset_index(drop=True)
#df = df[df.num_objs==1]

#df = df[df['ann_label']=='?'].reset_index(drop=True)
print(diagnosis)
print(df)

vis_level = 0
dim = (512, 512)

custom_field_map = {
        'SVS_ID':'string',
        'top_left': 'int list',
        'center': 'int list',
        'dim': 'int list',
        'vis_level': 'int',
        'diagnosis': 'string',
        'annotation_label':'string',
        'mask': 'double matrix'}

df_list = []

for SVS_ID in df.SVS_ID.unique():
    h_e_object = openslide.open_slide('/home/dgs2/data/DigitalPathologyAI/{}.svs'.format(SVS_ID))
    #gt_df = pd.read_csv('/home/dgs1/data/DigitalPathologyAI/MitoticDetection/omero/{}/{}_MF_ROIs.csv'.format(diagnosis, SVS_ID))
    gt_df = df[df['SVS_ID']==SVS_ID].reset_index(drop=True)
    print(gt_df)
    #gt_df['index'] = gt_df.index.to_list()
    masks = np.load('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/DetectionResults/{}_detected_masks.npy'.format(SVS_ID),allow_pickle=True)
    #masks = np.load( '/home/dgs2/data/DigitalPathologyAI/MitoticDetection/masks/{}_masks.npy'.format(SVS_ID), allow_pickle=True)

    masks = masks[gt_df['index']]
    #gt_df.reset_index(drop=True,inplace=True)

    nrrd_list = []

    for i in range(gt_df.shape[0]):
        top_left = np.array([gt_df.coords_x[i], gt_df.coords_y[i]]).astype('int')
        top_left = (top_left[0]-256,top_left[1]-256)
        #center = np.array([gt_df.centers_x[i], gt_df.centers_y[i]]).astype('int')
        h_e = np.array(h_e_object.read_region(top_left, vis_level, dim).convert("RGB"))
        #annotation_label = gt_df.ann_label[i]
        annotation_label = '?'
        mask = masks[i, :, :].astype('float')
        mask = np.concatenate((np.zeros_like(mask),mask),axis=0)
        mask = np.concatenate((np.zeros_like(mask), mask), axis=1)

        mask = mask / np.max(mask)
        masked = np.ma.masked_greater_equal(mask, 0.5)
        mask = masked.mask.astype(np.uint8)
        mask = np.array(255 * mask)

        bbox = get_bbox_from_mask(mask)

        center_x = int((bbox[0] + bbox[1]) / 2) + gt_df.coords_x[i]
        center_y = int((bbox[2] + bbox[3]) / 2) + gt_df.coords_y[i]
        center   = (center_x,center_y)


        header = {'SVS_ID': SVS_ID,
                  'top_left': top_left,
                  'center': center,
                  'dim': dim,
                  'vis_level': vis_level,
                  'diagnosis': diagnosis,#gt_df.diagnosis[i],
                  'source': source,
                  'annotation_label':annotation_label,
                  'mask': mask}

        nrrd.write('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/nrrd/{}_MF{}.nrrd'.format(SVS_ID, i), h_e, header, custom_field_map=custom_field_map)
        nrrd_list.append('{}_MF{}.nrrd'.format(SVS_ID, i))

        if i%100 == 0:
            print('{} files saved'.format(i))

    gt_df['nrrd_file'] = nrrd_list
    gt_df['ann_label'] = [annotation_label]*gt_df.shape[0]
    #gt_df.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/omero/{}/{}_MF_ROIs.csv'.format(diagnosis, SVS_ID),index=False)
    df_list.append(gt_df)

    print('NRRD for {} SAVED'.format(SVS_ID))

df = pd.concat(df_list)
df.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/omero/{}_MF_ROIs.csv'.format(diagnosis),index=False)







