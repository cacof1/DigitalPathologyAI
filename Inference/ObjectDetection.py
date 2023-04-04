# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:43:52 2022

@author: zhuoy
"""
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import toml
from datetime import datetime, date
import time
import pathlib

import torchvision.transforms as T
from QA.Normalization.Colour import ColourNorm_old
from PreProcessing.PreProcessingTools import PreProcessor
from Model.MaskRCNN import MaskFRCNN
from Model.MaskRCNN import MaskFRCNN

from Dataloader.Dataloader import *
from Utils import MultiGPUTools
import pytorch_lightning as pl
from Utils.PredsAnalyzeTools import Preds2Results

config = toml.load('/home/dgs2/Software/DigitalPathologyAI/Configs/MaskRCNN_Inference_config.ini')
n_gpus = 1#torch.cuda.device_count()


if config['DATA']['With_GroundTruth']:
    print('Inference on the testing dataset')
    tile_dataset = pd.read_csv(config['DATA']['Dataframe'])
    tile_dataset['SVS_ID'] = tile_dataset['SVS_ID'].astype('str')
    tile_dataset = tile_dataset[tile_dataset['SVS_ID'].isin(config['CRITERIA']['id_internal'])].reset_index(drop=True)

else:
    print('Inference on the other slides')
    SVS_dataset = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/Inference_SFT_intermediate.csv')
    #SVS_dataset.to_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/Inference_SFT_high.csv', index=False)
    #SVS_dataset = SVS_dataset.iloc[24:,:].reset_index(drop=True)
    #SVS_dataset = QueryImageFromCriteria(config)
    #SynchronizeSVS(config, SVS_dataset)
    DownloadNPY(config, SVS_dataset)

    SVS_PATH_list = []
    NPY_PATH_list = []
    for i in range(SVS_dataset.shape[0]):
        SVS_PATH_list.append('{}/{}.svs'.format(config['DATA']['SVS_Folder'], SVS_dataset['id_external'][i]))
        NPY_PATH_list.append('{}/patches/{}.npy'.format(config['DATA']['SVS_Folder'], SVS_dataset['id_external'][i]))
    SVS_dataset['SVS_PATH'] = SVS_PATH_list
    SVS_dataset['NPY_PATH'] = NPY_PATH_list

    print(SVS_dataset)
    print(SVS_dataset['diagnosis'].value_counts())

    print('Loading file parameters...', end='')

    error_list = []
    for count, SVS_ID in enumerate(SVS_dataset['id_internal'].unique()):
        SVS_dataset_temp = SVS_dataset[SVS_dataset['id_internal']==SVS_ID]
        try:
            tile_dataset_temp = LoadFileParameter(config, SVS_dataset_temp)
        except Exception as e:
            print(str(e))
            error_list.append(SVS_ID)

    SVS_dataset = SVS_dataset[~SVS_dataset['id_internal'].isin(error_list)]
    tile_dataset = LoadFileParameter(config, SVS_dataset)

    tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_Tumour'] > 0.85]  # keep only tumour tiles
    print('Done.')
    print(tile_dataset)
    tile_dataset['num_objs'] = [0] * tile_dataset.shape[0]

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = T.Compose([
    T.ToTensor(),  # this also normalizes to [0,1].
    ColourNorm_old.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#%%
thresh_num = 10000
for count, SVS_ID in enumerate(tile_dataset['id_external'].unique()):
    slide_dataset = tile_dataset[tile_dataset['id_external'] == SVS_ID].reset_index(drop=True)
    slide_dataset['SVS_PATH'] = [config['DATA']['SVS_Folder'] + '{}.svs'.format(SVS_ID)]*slide_dataset.shape[0]
    p = pathlib.Path(config['DATA']['Detection_Path'] + '{}_detected_coords.csv'.format(SVS_ID))
    if os.path.isfile(p):
        delta = datetime.now() - datetime.fromtimestamp(p.stat().st_ctime)
        time_difference = delta.days
    else:
        time_difference = 9999

    if time_difference < 5:
        print('{}: {} is already compeleted on {}'.format(count, SVS_ID, datetime.fromtimestamp(p.stat().st_ctime)))
        continue

    elif slide_dataset.shape[0] < thresh_num:#slide_dataset.shape[0] > 0:
        print('Processing {}: {}'.format(count, SVS_ID))

        data = DataLoader(DataGenerator(slide_dataset, transform=val_transform, inference=True),
                          batch_size=config['BASEMODEL']['Batch_Size'],
                          num_workers=config['MODEL']['num_of_workers'],
                          persistent_workers=True,
                          shuffle=False,
                          pin_memory=True,)

        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             benchmark=True,
                             callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

        model = MaskFRCNN(config).load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
        model.eval()

        predictions = trainer.predict(model, data)
        df, detected_masks = Preds2Results(predictions, slide_dataset,
                                           batch_size=config['BASEMODEL']['Batch_Size'],
                                           Detection_Path=config['DATA']['Detection_Path'],
                                           threshold=0.5,
                                           label_name='num_objs',
                                           save_name='detected')
    else:
        n = int(slide_dataset.shape[0] / thresh_num)
        for i in range(n+1):
            p = pathlib.Path(config['DATA']['Detection_Path'] + '{}_detected_{}_coords.csv'.format(SVS_ID,i))
            if os.path.isfile(p):
                print('{}: {}_{} is already compeleted'.format(count, SVS_ID, i))
                df_all = pd.read_csv(config['DATA']['Detection_Path'] + '{}_detected_coords.csv'.format(SVS_ID))
                detected_masks_all = np.load(config['DATA']['Detection_Path'] + '{}_detected_masks.npy'.format(SVS_ID))
                continue
            else:
                print('Processing {}: {}_{}/{}'.format(count, SVS_ID, i+1, n+1))
                data = DataLoader(DataGenerator(slide_dataset.iloc[i*thresh_num:(i+1)*thresh_num,:].reset_index(drop=True), transform=val_transform, inference=True),
                                  batch_size=config['BASEMODEL']['Batch_Size'],
                                  num_workers=0,  # config['MODEL']['num_of_workers'],
                                  # persistent_workers=True,
                                  shuffle=False,
                                  pin_memory=True, )

                trainer = pl.Trainer(accelerator='gpu', devices=1,
                                     benchmark=True,
                                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

                model = MaskFRCNN(config).load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
                model.eval()

                predictions = trainer.predict(model, data)
                df, detected_masks = Preds2Results(predictions, slide_dataset.iloc[i*thresh_num:(i+1)*thresh_num,:].reset_index(drop=True),
                                                   batch_size=config['BASEMODEL']['Batch_Size'],
                                                   Detection_Path=config['DATA']['Detection_Path'],
                                                   threshold=0.5,
                                                   label_name='num_objs',
                                                   save_name='detected_{}'.format(i))

            if i == 0:
                df_all = df
                detected_masks_all = detected_masks
            else:
                df_all = pd.concat([df_all,df])
                detected_masks_all = np.concatenate([detected_masks_all,detected_masks],axis=0)
                df_all.to_csv(config['DATA']['Detection_Path'] + '{}_detected_coords.csv'.format(slide_dataset['id_external'][0]),
                          index=False)
                np.save(config['DATA']['Detection_Path'] + '{}_detected_masks.npy'.format(slide_dataset['id_external'][0]),
                        detected_masks_all)
