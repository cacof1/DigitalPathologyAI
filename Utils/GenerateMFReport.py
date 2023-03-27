import numpy as np
import pandas as pd
import openslide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import seaborn
from scipy import stats
from Utils.PredsAnalyzeTools import *


def plot_bbox(df,HE_Path='/home/dgs2/data/DigitalPathologyAI/'):
    vis_level = 0
    dim = (256, 256)
    for i in range(df.shape[0]):
        SVS_ID = df['SVS_ID'][i]
        he_object = openslide.open_slide(HE_Path+'{}.svs'.format(SVS_ID))
        top_left = (df['coords_x'][i], df['coords_y'][i])
        img = np.array(he_object.read_region(top_left, vis_level, dim).convert("RGB"))
        fig, ax = plt.subplots()
        ax.imshow(img)
        #x = int(df['cell_x'][i]-df['coords_x'][i])
        #y = int(df['cell_y'][i]-df['coords_y'][i])
        #rect1 = patches.Rectangle((x,y), 64, 64, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((int(df['xmin'][i]),int(df['ymin'][i])), int(df['xmax'][i]-df['xmin'][i]), int(df['ymax'][i]-df['ymin'][i]), linewidth=2, edgecolor='b', facecolor='none')
        #ax.add_patch(rect1)
        ax.add_patch(rect2)
        plt.title('{}, cls: {}, detect:{}'.format(SVS_ID,df['prob_1'][i],df['scores'][i]))
        plt.show()

def CreateDensityMap(data, SVS_ID, HE_Path='/home/dgs2/data/DigitalPathologyAI/',if_plot=False):

    if data.shape[0] == 0:
        coord = (-9999, -9999)
        data_in = data
    elif data.shape[0] < 3:
        coord = (int((data.xmin[0] + data.xmax[0])/2 + data['coords_x'][0]), int((data.ymin[0] + data.ymax[0])/2 + data['coords_y'][0]))
        data_in = data
    else:
        vis_level = 0
        wsi_object = openslide.open_slide(HE_Path + '{}.svs'.format(SVS_ID))
        data['x_center'] = ((data.xmin + data.xmax)/2 + data['coords_x']) / wsi_object.level_downsamples[vis_level]
        data['y_center'] = ((data.ymin + data.ymax)/2 + data['coords_y']) / wsi_object.level_downsamples[vis_level]
        x = data.x_center
        y = data.y_center

        values = np.vstack([x, y])
        kde = stats.gaussian_kde(values,bw_method=10)
        density = kde(values)
        xy_max = values.T[np.argmax(density)]

        r = 3750
        region_size = (2 * r, 2 * r)
        center = (xy_max[0] * wsi_object.level_downsamples[vis_level], xy_max[1] * wsi_object.level_downsamples[vis_level])
        coord_x = int(center[0] - r)
        coord_y = int(center[1] - r)
        coord = (coord_x, coord_y)

        data['x_center_in'] = x * wsi_object.level_downsamples[vis_level] - coord_x
        data['y_center_in'] = y * wsi_object.level_downsamples[vis_level] - coord_y
        data_in = data[data.x_center_in > 0]
        data_in = data_in[data_in.y_center_in > 0]
        data_in = data_in[data_in.x_center_in < region_size[0]]
        data_in = data_in[data_in.y_center_in < region_size[1]]
        data_in['distance^2'] = (data_in.x_center_in - r) ** 2 + (data_in.y_center_in - r) ** 2
        data_in = data_in[data_in['distance^2'] < r ** 2].reset_index(drop=True)

        if if_plot:
            img = np.array(wsi_object.read_region((0, 0), vis_level, wsi_object.level_dimensions[vis_level]).convert("RGB"))
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.scatter(x, y, marker=".", color='red', s=1)
            plt.imshow(img)
            plt.title('{} Mitoses distribution'.format(SVS_ID))
            plt.axis('off')
            plt.show()

            fig, ax = plt.subplots(1, figsize=(9, 9))
            kde_map = seaborn.kdeplot(data=data, x="x_center", y="y_center", alpha=0.3, cmap='inferno', n_levels=50, shade=True,
                                  # hue='labels', cbar = True
                                  )
            p = kde_map.collections[-1].get_paths()[0]
            v = p.vertices
            lx = [v[r][0] for r in range(len(v))]
            ly = [v[r][1] for r in range(len(v))]
            plt.axvline(xy_max[0], c='r', lw=1)
            plt.axhline(xy_max[1], c='r', lw=1)
            plt.text(xy_max[0], xy_max[1],
                     f" x={xy_max[0] * wsi_object.level_downsamples[vis_level]:.2f}\n y={xy_max[1] * wsi_object.level_downsamples[vis_level]:.2f}",
                     color='black', ha='left', va='bottom', fontsize=14)
            plt.axis('off')
            plt.title('{} Kernel density estimation map'.format(SVS_ID))
            plt.imshow(img)
            plt.show()

    return coord, data_in

def Plot10HPFs(SVS_ID,coord,data_in,r = 3750, HE_Path='/home/dgs2/data/DigitalPathologyAI/'):
    vis_level = 0
    wsi_object = openslide.open_slide(HE_Path + '{}.svs'.format(SVS_ID))
    region_size = (2 * r, 2 * r)
    hpf_10 = np.array(wsi_object.read_region(coord, vis_level, region_size).convert("RGB"))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(hpf_10)
    draw_circle = plt.Circle((r, r), r, color='red', fill=False)
    ax.set_aspect(1)
    ax.add_artist(draw_circle)
    ax.scatter(data_in.x_center_in, data_in.y_center_in, marker=".", color='red', s=200)
    plt.axis('off')
    plt.title('{} Densest Region of Mitotic Activity (10 HPFs)'.format(SVS_ID))
    plt.text(r / 2, r / 2, f" Number of mitoses: {data_in.shape[0]:.0f}",
             color='black', ha='left', va='bottom',
             fontsize=18, weight='bold', fontstyle='oblique')
    plt.show()

def ReportMF(df, detect_thres=0.8, cls_thresh=0.5,
             HE_Path = '/home/dgs2/data/DigitalPathologyAI/',
             NPYPath = '/home/dgs2/data/DigitalPathologyAI/patches/',
             if_plot=False):

    SVS_IDs = []
    center_x = []
    center_y = []
    MF_10HPFs = []
    num_of_MFs = []
    num_of_tumour_tiles = []

    for SVS_ID in df.SVS_ID.unique():
        data = np.load(NPYPath + SVS_ID + '.npy', allow_pickle=True).tolist()
        coord_df = list(data.values())[0][-1]

        print('Generating MF report for Slide {}'.format(SVS_ID))
        data = df[df['SVS_ID'] == SVS_ID].reset_index(drop=True)
        data = data[data['prob_1'] > cls_thresh].reset_index(drop=True)
        data = data[data['scores'] > detect_thres].reset_index(drop=True)
        print(data.shape[0])
        try:
            coord, data_in = CreateDensityMap(data, SVS_ID, HE_Path, if_plot)
        except:
            print('Errors in {}'.format(SVS_ID))
            continue

        if if_plot:
            Plot10HPFs(SVS_ID, coord, data_in, HE_Path=HE_Path)
            plot_bbox(data_in, HE_Path)

        SVS_IDs.append(SVS_ID)
        center_x.append(coord[0] + 3750)
        center_y.append(coord[1] + 3750)
        MF_10HPFs.append(data_in.shape[0])
        num_of_MFs.append(data.shape[0])
        num_of_tumour_tiles.append(coord_df.shape[0])

    df_MF = pd.DataFrame()
    df_MF['SVS_ID'] = np.array(SVS_IDs)
    df_MF['num_of_tumour_tiles'] = np.array(num_of_tumour_tiles)
    df_MF['num_of_MFs'] = np.array(num_of_MFs)
    df_MF['average_density_10HPFs'] = round(df_MF.loc[:,'num_of_MFs']/df_MF.loc[:,'num_of_tumour_tiles'] * 674,3)
    df_MF['highest_density_10HPFs'] = np.array(MF_10HPFs)
    df_MF['center_x'] = np.array(center_x)
    df_MF['center_y'] = np.array(center_y)

    return df_MF

#%%
HE_Path = '/home/dgs2/data/DigitalPathologyAI/'
NPYPath = '/home/dgs2/data/DigitalPathologyAI/patches/'
Detection_Path = '/home/dgs2/data/DigitalPathologyAI/MitoticDetection/DetectionResults/'
diagnosis = 'SFT_high'
#SFT_intermediate
#desmoid_fibromatosis
#synovial_sarcoma
#superficial_fibromatosis
version = 0
df = pd.read_csv(Detection_Path + 'classification_coords_{}{}.csv'.format(diagnosis,version),low_memory=False)
df.rename(columns={'prob_yes':'prob_1','prob_no':'prob_0'},inplace=True)
df['SVS_ID'] = df['SVS_ID'].astype('str')
df = df[df['prob_tissue_type_Tumour']>0.94].reset_index(drop=True)

df_MF = ReportMF(df, detect_thres=0.8, cls_thresh=0.5, HE_Path=HE_Path, NPYPath=NPYPath)
df_MF.to_csv(Detection_Path + '{}_MF_Report{}.csv'.format(diagnosis,version),index=False)
print('{}_MF_Report{}.csv Saved'.format(diagnosis,version))
