import pandas as pd
import numpy as np

df_list = []

for diagnosis in ['SFT_high','desmoid_fibromatosis','superficial_fibromatosis','SFT_low','synovial_sarcoma']:
    df = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/omero/{}_MF_ROIs.csv'.format(diagnosis))
    df['diagnosis'] = [diagnosis]*df.shape[0]
    df.rename(columns={'prob_tissue_type_tumour': 'prob_tissue_type_Tumour'}, inplace=True)
    df_list.append(df)

df_ann = pd.concat(df_list)
#df.to_csv('/home/dgs1/data/DigitalPathologyAI/MitoticDetection/omero/AllMF_25102022.csv',index=False)

df_ann.ann_label.value_counts()

df_pHH3 = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/omero/pHH3_MF_ROIs.csv')
df_pHH3.rename(columns={'prob_tissue_type_tumour':'prob_tissue_type_Tumour'},inplace=True)
df_pHH3['ann_label'] = ['yes']*df_pHH3.shape[0]

df_ann = df_ann.loc[:,['SVS_ID','coords_x', 'coords_y','prob_tissue_type_Tumour','ann_label','nrrd_file','diagnosis']]
df_pHH3 = df_pHH3.loc[:,['SVS_ID','coords_x', 'coords_y','prob_tissue_type_Tumour','ann_label','nrrd_file','diagnosis']]

df_ann['source'] = ['AI']*df_ann.shape[0]
df_pHH3['source'] = ['pHH3']*df_pHH3.shape[0]

print(df_ann)
print(df_pHH3)
df_all = pd.concat([df_ann,df_pHH3])

df_all.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_24022023.csv',index=False)
print('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/AllMF_24022023.csv Saved')



