import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import toml
from datetime import datetime, date
import pandas as pd
import time
import pathlib
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.MaskRCNN import MaskFRCNN
from Utils.OmeroTools import (
    QueryImageFromCriteria,
    SynchronizeSVS,
    SynchronizeNPY
)
from Dataloader.Dataloader import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from Utils.PredsAnalyzeTools import Preds2Results

def load_config(config_file):
    return toml.load(config_file)

def get_tile_dataset(config):
    SVS_dataset = QueryImageFromCriteria(config)
    SynchronizeSVS(config, SVS_dataset)
    SynchronizeNPY(config, SVS_dataset)
    tile_dataset = LoadFileParameter(config, SVS_dataset)
    tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_Tumour'] > 0.94]
    tile_dataset = tile_dataset.merge(SVS_dataset, on='id_external')
    tile_dataset['SVS_PATH'] = tile_dataset['SVS_PATH_y'] # Ugly
    return tile_dataset#, tile_dataset_full, valid_tumour_tiles_index

def get_transforms(config):
    if config['QC']['macenko_norm']:
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # this also normalizes to [0,1].
            ColourNorm.Macenko(saved_fit_file=config['QC']['macenko_file']) if 'macenko_file' in config['QC'] else None,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),  # this also normalizes to [0,1].
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return val_transform

def Inference(config_file):
    config = toml.load(config_file)
    n_gpus = 1#torch.cuda.device_count()

    SVS_dataset = QueryImageFromCriteria(config)
    SynchronizeNPY(config, SVS_dataset)
    tile_dataset = get_tile_dataset(config)
    print(tile_dataset['id_external'].value_counts())

    for count, SVS_ID in enumerate(tile_dataset['id_external'].unique()):
        slide_dataset = tile_dataset[tile_dataset['id_external'] == SVS_ID].reset_index(drop=True)
        print(slide_dataset)

        p = pathlib.Path(config['DATA']['Detection_Path'] + '{}_detected_coords.csv'.format(SVS_ID))
        if os.path.isfile(p):
            delta = datetime.now() - datetime.fromtimestamp(p.stat().st_ctime)
            time_difference = delta.days
        else: time_difference = 9999

        if time_difference < 5:
            print('{}: {} is already compeleted on {}'.format(count, SVS_ID, datetime.fromtimestamp(p.stat().st_ctime)))
            continue

        print('Processing {}: {}'.format(count, SVS_ID))

        data = DataLoader(DataGenerator(slide_dataset, config, transform=get_transforms(config)),
                          batch_size=config['BASEMODEL']['Batch_Size'],
                          num_workers=config['MODEL']['num_of_workers'],
                          persistent_workers=True,
                          shuffle=False,
                          pin_memory=True,)

        trainer = L.Trainer(accelerator='gpu', devices=1,
                             benchmark=True,
                             callbacks=[TQDMProgressBar(refresh_rate=1)])

        model = MaskFRCNN(config).load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
        model.eval()

        predictions = trainer.predict(model, data)
        df, detected_masks = Preds2Results(predictions, slide_dataset,
                                           batch_size=config['BASEMODEL']['Batch_Size'],
                                           Detection_Path=config['DATA']['Detection_Path'],
                                           threshold=0.5,
                                           #label_name='num_objs',
                                           save_name='detected')

if __name__ == "__main__":
    Inference(sys.argv[1])