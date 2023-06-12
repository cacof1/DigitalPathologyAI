import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import toml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.MitoticConvNet import ConvNet
from Dataloader.ObjectDetection import *
import pandas as pd
import numpy as np
from Dataloader.Dataloader import *
import torch
from Utils.OmeroTools import (
    QueryImageFromCriteria,
    SynchronizeSVS,
    SynchronizeNPY
)

def load_config(config_file):
    return toml.load(config_file)

def get_tile_dataset(config):
    try:
        slidedataset = pd.read_csv(config['DATA']['Detection_Path'] + '{}_detected_coords.csv'.format(SVS_ID))
    except:
        print('{} Not Found'.format(SVS_ID))
    slidedataset['SVS_ID'] = [SVS_ID] * slidedataset.shape[0]
    slidedataset['index'] = slidedataset.index
    print(slidedataset)

    return slidedataset

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

    n_gpus = 1
    config = toml.load(config_file)

    SVS_dataset = QueryImageFromCriteria(config)
    SynchronizeNPY(config, SVS_dataset)

    model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
    model.eval()
    print('Model loaded: {}'.format(config['CHECKPOINT']['Model_Save_Path']))

    df_list = []
    for count, SVS_ID in enumerate(SVS_dataset['id_external'].unique()):
        print('Processing {}/{}: {}'.format(count, len(SVS_dataset['id_external'].unique()), SVS_ID))
        slidedataset = get_tile_dataset(config)

        data = DataLoader(MixDataset(slidedataset,
                                     masked_input=config['DATA']['masked_input'],
                                     wsi_folder=config['DATA']['SVS_Folder'],
                                     mask_folder=config['DATA']['Mask_Folder'],
                                     data_source=config['DATA']['data_source'],
                                     dim=config['BASEMODEL']['Patch_Size'],
                                     vis_level=0,
                                     channels=3,
                                     transform=val_transform,
                                     inference=True),
                          num_workers=config['BASEMODEL']['Num_of_Worker'],
                          persistent_workers=True,
                          batch_size=config['BASEMODEL']['Batch_Size'],
                          shuffle=False,
                          pin_memory=True,)

        trainer = L.Trainer(accelerator='gpu',devices=1,
                             #strategy='ddp_find_unused_parameters_false',
                             benchmark=True,
                             precision=config['BASEMODEL']['Precision'],
                             callbacks=[TQDMProgressBar(refresh_rate=1)])

        predictions = trainer.predict(model, data)
        predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))

        classes = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))
        for i, class_name in enumerate(classes):
            slidedataset['prob_' + str(class_name)] = pd.Series(predicted_classes_prob[:, i], index=slidedataset.index)

        df_list.append(slidedataset)

    df_all = pd.concat(df_list)
    df_all.to_csv(config['DATA']['Detection_Path'] + 'classification_coords_{}_{}.csv'.format(config['DATA']['SaveName'], config['DATA']['Version']), index=False)

    print('Done.')

if __name__ == "__main__":
    Inference(sys.argv[1])











