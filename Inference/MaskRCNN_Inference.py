# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:43:52 2022

@author: zhuoy
"""
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import toml
import torchvision.transforms as T
from QA.Normalization.Colour import ColourNorm
from PreProcessing.PreProcessingTools import PreProcessor
from Model.MaskRCNN_Model import MaskFRCNN

from Dataloader.Dataloader import *
from Utils import MultiGPUTools
import pytorch_lightning as pl


def CreateTileDataset(dataset, prob_label='prob_tissue_type_tumour', threshold=0.8):

    tile_dataset = pd.DataFrame()
    for npy_path in dataset.NPY_PATH:
        data = np.load(npy_path, allow_pickle=True).item()
        key = list(data.keys())[0]
        header, existing_df = data[key]
        tile_dataset = pd.concat([tile_dataset, existing_df], ignore_index=True)
        tile_dataset = tile_dataset[tile_dataset[prob_label] > threshold]
        tile_dataset.reset_index(inplace=True, drop=True)

    return tile_dataset

n_gpus = torch.cuda.device_count()

config = toml.load('E:/Projects/DigitalPathologyAI/Configs/MaskRCNN_Inference_config.ini')

SVS_dataset = QueryFromServer(config)
SynchronizeSVS(config, SVS_dataset)
print(SVS_dataset)
SVS_dataset.reset_index(inplace=True,drop=True)

tile_dataset = CreateTileDataset(SVS_dataset)
n_pad = MultiGPUTools.pad_size(len(tile_dataset), n_gpus, config['BASEMODEL']['Batch_Size'])
tile_dataset = MultiGPUTools.pad_dataframe(tile_dataset, n_pad)
#%%
pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = T.Compose([
    T.ToTensor(),  # this also normalizes to [0,1].
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = DataLoader(DataGenerator(tile_dataset, transform=val_transform, svs_folder=config['DATA']['SVS_Folder'], inference=True),
                  batch_size=config['BASEMODEL']['Batch_Size'],
                  num_workers=config['BASEMODEL']['Num_of_Workers'],
                  #persistent_workers=True,
                  shuffle=False,
                  pin_memory=True)

trainer = pl.Trainer(gpus=n_gpus,
                     #strategy='ddp',
                     benchmark=True,
                     precision=config['BASEMODEL']['Precision'],
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

model = MaskFRCNN(config).load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
model.eval()

predictions = trainer.predict(model, data)
np.save('E:/Projects/DigitalPathologyAI/maskrcnn_predictions.npy', predictions)

ordered_preds = MultiGPUTools.reorder_predictions(predictions)  # reorder if processing was done on multiple GPUs
reordered_preds = torch.Tensor.cpu(torch.cat(ordered_preds))

# Drop padding
tile_dataset = tile_dataset.iloc[:-n_pad]
predicted_classes_prob = reordered_preds[:-n_pad]