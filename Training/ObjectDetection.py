
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import toml
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from QA.Normalization.Colour import ColourNorm_old
import albumentations as A
from Utils.ObjectDetectionTools import collate_fn
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from Dataloader.ObjectDetection import MFDataModule
from Model.MaskRCNN import MaskFRCNN
from sklearn.model_selection import train_test_split


config = toml.load('/home/dgs2/Software/DigitalPathologyAI/Configs/MaskRCNN_config.ini')
n_gpus = len(config['MODEL']['GPU_ID'])

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

df = pd.read_csv(config['DATA']['dataframe'])
df = df[(df['quality'] == 1) & (df['refine'] == 0)]

df = df[df['ann_label'] == 'yes']
#df = df[df['ann_label'] != '?']

print(df)

df_test = df[df['SVS_ID'].isin(config['DATA']['filenames_test'])]
df_test.reset_index(drop=True, inplace=True)

df_all = df[~df['SVS_ID'].isin(config['DATA']['filenames_test'])]
filenames = list(df_all.SVS_ID.unique())
train_idx, val_idx = train_test_split(filenames, test_size=config['DATA']['val_size'], train_size=config['DATA']['train_size'])
test_idx = list(df_test.SVS_ID.unique())

print('{} Train slides: {}'.format(len(train_idx),train_idx))
print('{} Val slides: {}'.format(len(val_idx),val_idx))
print('{} Test slides: {}'.format(len(test_idx),test_idx))

df_train = df_all[df_all['SVS_ID'].isin(train_idx)]
#df_train = df_all[df_all['SVS_ID'].isin(config['DATA']['filenames_train'])]
df_train.reset_index(drop=True, inplace=True)

df_val = df_all[df_all['SVS_ID'].isin(val_idx)]
#df_val = df_all[df_all['SVS_ID'].isin(config['DATA']['filenames_val'])]
df_val.reset_index(drop=True, inplace=True)


dm = MFDataModule(df_train, 
                  df_val, 
                  df_test,  
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

train_size = len(dm.train_data)
val_size = len(dm.val_data)
test_size = len(dm.test_data)
print('Training Size: {0}/{3}({4})\nValidating Size: {1}/{3}({5})\nTesting Size: {2}/{3}({6})'.format(train_size,
                                                                                                      val_size,
                                                                                                      test_size,
                                                                                                      df.shape[0],
                                                                                                      train_size /
                                                                                                      df.shape[0],
                                                                                                      val_size /
                                                                                                      df.shape[0],
                                                                                                      test_size /
                                                                                                      df.shape[0],
                                                                                                      ))


#%%
seed_everything(config['MODEL']['random_seed'])
 
model = MaskFRCNN(config)

checkpoint_callback = ModelCheckpoint(
    monitor=config['CHECKPOINT']['monitor'],
    dirpath=os.path.join(config['MODEL']['log_path'],config['MODEL']['base_model']),
    filename=config['CHECKPOINT']['filename'],
    save_top_k=config['CHECKPOINT']['save_top_k'],
    mode=config['CHECKPOINT']['mode'],
)

logger = TensorBoardLogger(config['MODEL']['log_path'], name=config['MODEL']['base_model'])

trainer = Trainer(accelerator="gpu",
                  devices=config['MODEL']['GPU_ID'],
                  #strategy='ddp_find_unused_parameters_false',
                  benchmark=True,
                  max_epochs=config['MODEL']['max_epochs'],
                  callbacks=[checkpoint_callback],
                  logger=logger,)

trainer.fit(model,dm)

