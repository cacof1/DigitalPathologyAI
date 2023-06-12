from Dataloader.Dataloader import *
from Dataloader.ObjectDetection import *
import toml
import sys
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
from torchvision import transforms
import torch
from QA.Normalization.Colour import ColourNorm
from QA.Normalization.Colour import ColourAugment
from Model.ConvNet import ConvNet
from sklearn import preprocessing
from Utils.OmeroTools import (
    QueryImageFromCriteria,
    SynchronizeSVS,
    SynchronizeNPY
)
import os
from datetime import datetime

def load_config(config_file):
    return toml.load(config_file)

def get_tile_dataset(config):
    df = pd.read_csv(config['DATA']['Dataframe'])
    df_error = pd.read_csv(config['DATA']['Errors_df'])
    df = df[~df['nrrd_file'].isin(df_error['nrrd_file'])]
    df = df.astype({'SVS_ID': 'str'})
    df = df[(df['quality'] == 1) & (df['refine'] == 0)]
    df = df[df['ann_label'] != '?']

    le = preprocessing.LabelEncoder()
    le.fit(df['ann_label'])
    df['ann_label'] = le.transform(df['ann_label'])
    config['DATA']['N_Classes'] = len(df['ann_label'].unique())
    df.reset_index(drop=True, inplace=True)
    print(df)

    df_train_val = df[~df['SVS_ID'].isin(config['DATA']['filenames_test'])]
    df_test = df[df['SVS_ID'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)
    test_idx = list(df_test.SVS_ID.unique())
    #print('{} Test slides: {}'.format(len(test_idx), test_idx))
    print('Testing Size: {}/{}({}) Positive Rate: {}'.format(len(df_test), len(df), len(df_test) / len(df), df_test['ann_label'].value_counts(normalize=True)[1]))

    return df_train_val, df_test, le

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
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        ColourAugment.ColourAugment(sigma=config['AUGMENTATION']['Colour_Sigma'], mode=config['AUGMENTATION']['Colour_Mode']),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def main(config_file):

    config = toml.load(config_file)
    now = datetime.now()
    timestamp = now.strftime("%m_%d_%Y_%H_%M")

    logger = get_logger(config, timestamp)
    callbacks = get_callbacks(config, timestamp)

    L.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)
    train_transform, val_transform = get_transforms(config)

    trainer = L.Trainer(devices=config['BASEMODEL']['GPU_ID'],
                        accelerator="gpu",
                        benchmark=False,
                        max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                        precision=config['BASEMODEL']['Precision'],
                        callbacks=callbacks,
                        logger=logger)

    df_train_val, df_test, le = get_tile_dataset(config)

    data = MFDataModule(df = df_train_val,
                        df_test = df_test,
                        val_size=config['DATA']['val_size'],
                        DataType='MixDataset',
                        masked_input=config['DATA']['masked_input'],
                        #wsi_folder=config['DATA']['SVS_Folder'],
                        #mask_folder=config['DATA']['mask_folder'],
                        nrrd_path = config['DATA']['nrrd_path'],
                        data_source = config['DATA']['data_source'],
                        patch_size = config['BASEMODEL']['Patch_Size'],
                        vis_list=[0],
                        batch_size=config['BASEMODEL']['Batch_Size'],
                        num_of_worker=config['BASEMODEL']['Num_of_Worker'],
                        train_transform=train_transform,
                        val_transform=val_transform,
                        extract_feature=False,
                        inference=False,)

    model = ConvNet(config, label_encoder=le)

    trainer.fit(model, data)
    trainer.test(ckpt_path='best', dataloaders=data.test_dataloader())

if __name__ == "__main__":
    main(sys.argv[1])
