# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:06:14 2022

@author: zhuoy
"""

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import toml
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from QA.Normalization.Colour import ColourNorm
import albumentations as A

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from Dataloader.ObjectDetection import MFDataModule
from Model.MaskRCNN import MaskFRCNN

config = toml.load(sys.argv[1])
n_gpus = torch.cuda.device_count()

augmentation = A.Compose([
    A.RandomCrop(width=config['DATA']['dim'][0], height=config['DATA']['dim'][1]),
    A.HorizontalFlip(p=config['AUGMENTATION']['horizontalflip']),
    #A.RandomBrightnessContrast(p=config['AUGMENTATION']['randombrightnesscontrast']),
])

if config['QC']['macenko_norm']:
    normalization = T.Compose([
        #T.ToTensor(),  # this also normalizes to [0,1].
        ColourNorm.Macenko(saved_fit_file=config['QC']['macenko_file']),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

else:
    normalization = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

df = pd.read_csv(config['DATA']['dataframe']) 
wsi_folder = config['DATA']['wsi_folder']
mask_folder = config['DATA']['mask_folder']

df_all = df[df['num_objs']==1] 

filenames = df_all.filename.value_counts().index.to_list()

#df_random_split = df_all[df_all['filename'].isin(config['DATA']['filenames_random_split'])]
#df_random_split = df_random_split.sample(frac=1.0)
#split_index_1 = int(df_random_split.shape[0] * 0.7)
#split_index_2 = int(df_random_split.shape[0] * 0.8)
#df_train = df_random_split.iloc[0: split_index_1, :]
#df_val = df_random_split.iloc[split_index_1:split_index_2, :]
#df_test = df_random_split.iloc[split_index_2:, :]

df_train = df_all[df_all['filename'].isin(config['DATA']['filenames_train'])]
#df_train = pd.concat([df_all[df_all['filename'].isin(config['DATA']['filenames_train'])],df_train],axis=0)
df_train.reset_index(drop = True, inplace = True)

df_val = df_all[df_all['filename'].isin(config['DATA']['filenames_val'])]
#df_val = pd.concat([df_all[df_all['filename'].isin(config['DATA']['filenames_val'])],df_val],axis=0)
df_val.reset_index(drop = True, inplace = True)

df_test = df_all[df_all['filename'].isin(config['DATA']['filenames_test'])]
#df_test = pd.concat([df_all[df_all['filename'].isin(config['DATA']['filenames_test'])],df_test],axis=0)
df_test.reset_index(drop = True, inplace = True)

dm = MFDataModule(df_train,#.iloc[:50,:],
                  df_val,#.iloc[:50,:],
                  df_test,#.iloc[:50,:],
                  wsi_folder,
                  mask_folder,
                  batch_size=config['MODEL']['batch_size'],
                  num_of_worker=config['MODEL']['num_of_worker'],
                  augmentation=augmentation, 
                  normalization=normalization,
                  inference=config['MODEL']['inference'])

train_size = len(dm.train_data)
val_size = len(dm.val_data)
test_size = len(dm.test_data)
print('Training Size: {0}/{3}({4})\nValidating Size: {1}/{3}({5})\nTesting Size: {2}/{3}({6})'.format(train_size,
                                                                                                      val_size,
                                                                                                      test_size,
                                                                                                      df_all.shape[0],
                                                                                                      train_size/df_all.shape[0],
                                                                                                      val_size/df_all.shape[0],
                                                                                                      test_size/df_all.shape[0],
                                                                                                      ))
'''
N = random.randint(0, train_size)
image,target = dm.train_data[N]

plt.subplot(1,2,1)
plt.imshow(image.permute(1, 2, 0)) 
plt.title('Image {}'.format(N))
plt.subplot(1,2,2)
plt.imshow(target['masks'].squeeze()) 
plt.show()
'''

#%%
seed_everything(config['MODEL']['random_seed'])
 
model = MaskFRCNN(config)

log_path = config['MODEL']['log_path']

checkpoint_callback = ModelCheckpoint(
    monitor=config['CHECKPOINT']['monitor'],
    dirpath=log_path,
    filename=config['CHECKPOINT']['filename'],
    save_top_k=config['CHECKPOINT']['save_top_k'],
    mode=config['CHECKPOINT']['mode'],
)

logger = TensorBoardLogger(log_path, name=config['MODEL']['saved_model_name'])

'''trainer = Trainer(gpus=1, 
                  max_epochs=config['MODEL']['max_epochs'],
                  callbacks=[checkpoint_callback],
                  logger=logger,)'''

trainer = Trainer(gpus=n_gpus,
                  strategy='ddp_find_unused_parameters_false',
                  benchmark=True,
                  max_epochs=config['MODEL']['max_epochs'],
                  callbacks=[checkpoint_callback],
                  logger=logger,
                  )

trainer.fit(model,dm)
trainer.save_checkpoint(config['MODEL']['saved_model_name'])

