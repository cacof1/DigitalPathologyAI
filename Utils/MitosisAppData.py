import pandas as pd
import numpy as np
import glob
import time
import os
import nrrd
custom_field_map = {
        'SVS_ID':'string',
        'top_left': 'int list',
        'center': 'int list',
        'dim': 'int list',
        'vis_level': 'int',
        'diagnosis': 'string',
        'annotation_label': 'string',
        'mask': 'double matrix'}

nrrd_path = '/home/dgs2/data/DigitalPathologyAI/MitoticDetection/nrrd/'
#%%
def visualize_prediction(filename):
    data, header = nrrd.read(os.path.join(nrrd_path,filename), custom_field_map)
    top_left = header['top_left']
    diagnosis = header['diagnosis']
    SVS_ID = header['SVS_ID']
    center = header['center']
    annotation_label = header['annotation_label']

    return top_left, diagnosis, SVS_ID, center, annotation_label

annotation_types = [
    "MF",
    'unknown',
    "FP",
    ]

users = ['pHH3','Eleanna','Adrienne','Bill','Rebecca','Final']

df_nrrd = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_24022023.csv')
df_nrrd = df_nrrd.rename(columns={"ann_label": "label"})
df_nrrd = df_nrrd.replace('yes', 'MF')
df_nrrd = df_nrrd.replace('no', 'FP')
df_nrrd = df_nrrd.replace('?', 'unknown')
df_nrrd = df_nrrd.drop(['coords_x','coords_y'],axis=1)

centers_x = []
centers_y = []

for i in range(df_nrrd.shape[0]):
    top_left, diagnosis, SVS_ID, center, annotation_label = visualize_prediction(df_nrrd['nrrd_file'][i])
    centers_x.append(center[0])
    centers_y.append(center[1])
    if i%100==0: print(i, center)

df_nrrd['center_x'] = centers_x
df_nrrd['center_y'] = centers_y
df_nrrd['annotator'] = 6060*['Eleanna'] + 1212*['AI'] + (df_nrrd.shape[0] - 6060 - 1212)*['pHH3']
df_nrrd['comment'] = np.nan
df_nrrd.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_24022023_v1.csv',index=False)
#%%
df_nrrd = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_24022023_v1.csv')
df = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/Annotations.csv')

for i in range(df.shape[0]):
    df_file = df_nrrd[df_nrrd['nrrd_file'] == df['index'][i]].iloc[0,:]
    new_row = pd.Series({'SVS_ID': df_file['SVS_ID'],
                         'prob_tissue_type_Tumour': df_file['prob_tissue_type_Tumour'],
                         'label': df['label'][i],
                         'nrrd_file': df_file['nrrd_file'],
                         'diagnosis': df_file['diagnosis'],
                         'center_x': df_file['center_x'],
                         'center_y': df_file['center_y'],
                         'annotator': df['user'][i],
                         'comment': df['comment'][i],
                         })

    df_nrrd = pd.concat([df_nrrd, new_row.to_frame().T], ignore_index=True)

df_nrrd = df_nrrd.sort_values(by=['diagnosis','SVS_ID', 'nrrd_file']).reset_index(drop=True)
df_nrrd.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_24022023_v2.csv',index=False)

#%%
'''def time_passed(start=0):
    return round(time.mktime(time.localtime())) - start

def dataframe_to_table_data(df):
    df = df.rename(columns={"nrrd_file": "Image ID",
                            "label": "Label",
                            "comment": "Comment",
                            "annotator": "Annotator",
                            })
    return df.loc[:, columns].to_dict('records')

def add_row_to_dataframe(df, label, comment, user):
    new_row = pd.Series({'SVS_ID': df.iloc[0, :]['SVS_ID'],
                         'prob_tissue_type_tumour': df.iloc[0, :]['prob_tissue_type_tumour'],
                         'label': label,
                         'nrrd_file': df.iloc[0, :]['nrrd_file'],
                         'diagnosis': df.iloc[0, :]['diagnosis'],
                         'center_x': df.iloc[0, :]['center_x'],
                         'center_y': df.iloc[0, :]['center_y'],
                         'annotator': user,
                         'comment': comment,
                         })
    df_user = df[df['annotator'] == user]
    if df_user.shape[0] > 0:
        df.loc[df_user.index.unique()[0]] = new_row
    else:
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    return df

columns = ['Image ID',"Label",'Comment','Annotator']
df_columns = ['nrrd_file','label','comment','annotator']
df_nrrd["timestamp"] = time_passed(0)

annotations_table_all_data = df_nrrd.to_dict('records')
annotations_df = pd.DataFrame.from_dict(annotations_table_all_data)
file = '492006_MF1.nrrd'
annotations_df = annotations_df[annotations_df['nrrd_file']==file]

annotations_table_data = dataframe_to_table_data(annotations_df)
annotations_df = add_row_to_dataframe(annotations_df, 'MF', 'SSS', 'Adrienne')

#%%

fig_shapes = [table_row_to_shape(tr, SVS_ID, center[0], center[1]) for tr in annotations_table_data]
new_shapes_i = []
old_shapes_i = []
for i, sh in enumerate(fig_shapes):
    if not shape_in(annotations_store_data[file]["shapes"])(sh):
        new_shapes_i.append(i)
    else:
        old_shapes_i.append(i)

for i in new_shapes_i:
    fig_shapes[i]["timestamp"] = time_passed(annotations_store_data["starttime"])

for i in old_shapes_i:
    old_shape_i = index_of_shape(
        annotations_store_data[file]["shapes"], fig_shapes[i]
    )
    fig_shapes[i]["timestamp"] = annotations_store_data[file]["shapes"][
        old_shape_i
    ]["timestamp"]

annotations_store_data[file]["shapes"] = fig_shapes
annotations_store_df = pd.DataFrame.from_dict(annotations_store_data)
annotations_table_all_data = annotations_store_df.loc[:,
                             ['nrrd_file', 'label', 'comment', 'annotator']].values.tolist()





#%%
def time_passed(start=0):
    return round(time.mktime(time.localtime())) - start

def update_default_store(df,default_ann):

    for file in df['nrrd_file'].unique():
        df_file = df[df['nrrd_file']==file].reset_index(drop=True)
        fig_shapes = []
        for i in range(df_file.shape[0]):
            fig_shape = {}
            for item in ['label','center_x','center_y', 'comment', 'SVS_ID', 'annotator']:
                fig_shape[item] = df_file[item][i]
            fig_shape['timestamp'] = time_passed(start=0)
            fig_shapes.append(fig_shape)

        default_ann[file]["shapes"] = fig_shapes

    return default_ann

def shape_to_table_row(label,index,comment,user):
    return {
        'Image ID': index,
        "Label": label,
        'Comment':comment,
        'Annotator':user
    }

df_nrrd = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_25102022_v2.csv')
default_ann = dict(**{file: {"shapes": []} for file in df_nrrd['nrrd_file'].to_list()},**{"starttime": time_passed()})

annotations_store_data = update_default_store(df_nrrd, default_ann)

key = '492006_MF1.nrrd'

annotations_table_all_data = []
for file in df_nrrd['nrrd_file'].to_list():
    for sh in annotations_store_data[file]["shapes"]:
        annotations_table_all_data.append(shape_to_table_row(sh['label'], file[:-5], sh['comment'], sh['annotator']))'''