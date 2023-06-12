
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
import toml
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from QA.Normalization.Colour import ColourNorm
import albumentations as A
from Utils.ObjectDetectionTools import collate_fn
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
from Dataloader.ObjectDetection import MFDataModule
from Model.MaskRCNN import MaskFRCNN
from sklearn.model_selection import train_test_split

def load_config(config_file):
    return toml.load(config_file)

def get_tile_dataset(config):
    df = pd.read_csv(config['DATA']['dataframe'])
    df = df[(df['quality'] == 1) & (df['refine'] == 0)]
    df = df[df['ann_label'] == 'yes']
    df_test = df[df['SVS_ID'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)
    df_train_val = df[~df['SVS_ID'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)

    test_idx = list(df_test.SVS_ID.unique())
    #print('{} Test slides: {}'.format(len(test_idx), test_idx))
    print('Testing Size: {}/{}({}) Positive Rate: {}'.format(len(df_test), len(df), len(df_test) / len(df), list(df_test['ann_label'].value_counts(normalize=True))[0]))

    return df_train_val, df_test

def get_logger(config, timestamp):
    return TensorBoardLogger(os.path.join(config['CHECKPOINT']['logger_folder'], config['CHECKPOINT']['model_name']),
                             name=timestamp)

def get_callbacks(config, timestamp):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath   = os.path.join(config['CHECKPOINT']['logger_folder'], config['CHECKPOINT']['model_name'], timestamp),
        monitor   = config['CHECKPOINT']['Monitor'],
        filename  = config['CHECKPOINT']['filename'],
        save_top_k= config['CHECKPOINT']['save_top_k'],
        mode      = config['CHECKPOINT']['Mode'])

    return [lr_monitor, checkpoint_callback]

def get_transforms(config):
    augmentation = A.Compose([
        A.RandomCrop(width=config['DATA']['dim'][0], height=config['DATA']['dim'][1]),
        A.HorizontalFlip(p=config['AUGMENTATION']['horizontalflip']),
        A.RandomBrightnessContrast(p=config['AUGMENTATION']['randombrightnesscontrast']),
    ])

    if config['QC']['macenko_norm']:
        normalization = T.Compose([
            ColourNorm_old.Macenko(saved_fit_file=config['QC']['macenko_file']),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        normalization = T.Compose([
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return augmentation, normalization

def main(config_file):

    config = toml.load(config_file)
    n_gpus = 1

    now = datetime.now()
    timestamp = now.strftime("%m_%d_%Y_%H_%M")

    augmentation, normalization = get_transforms(config)
    df_train_val, df_test = get_tile_dataset(config)

    data = MFDataModule(df = df_train_val,
                        df_test = df_test,
                        val_size=config['DATA']['val_size'],
                      data_source=config['DATA']['data_source'],
                      DataType='MFDataset',
                      wsi_folder=config['DATA']['wsi_folder'],
                      mask_folder=config['DATA']['mask_folder'],
                      nrrd_path=config['DATA']['nrrd_path'],
                      batch_size=config['MODEL']['batch_size'],
                      num_of_worker=config['MODEL']['num_of_worker'],
                      augmentation=augmentation,
                      normalization=normalization,
                      inference=config['MODEL']['inference'],
                      collate_fn=collate_fn)

    logger = get_logger(config, timestamp)
    callbacks = get_callbacks(config, timestamp)
    L.seed_everything(config['MODEL']['Random_Seed'], workers=True)

    model = MaskFRCNN(config)

    trainer = L.Trainer(devices=config['MODEL']['GPU_ID'],
                        accelerator="gpu",
                        benchmark=False,
                        max_epochs=config['MODEL']['Max_Epochs'],
                        callbacks=callbacks,
                        logger=logger)

    trainer.fit(model,data)
    trainer.test(ckpt_path='best', dataloaders=data.test_dataloader())

if __name__ == "__main__":
    main(sys.argv[1])
