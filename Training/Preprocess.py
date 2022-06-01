import os
from Dataloader.Dataloader import *
from PreProcessing.PreProcessingTools import PreProcessor
import toml
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet

config = toml.load(sys.argv[1])

########################################################################################################################
# 1. Download all relevant files based on the configuration file

dataset = QueryFromServer(config)
Synchronize(config, dataset)
print(dataset)

########################################################################################################################
# 2. Pre-processing: create npy files

preprocessor = PreProcessor(config)
coords_file = preprocessor.getTilesFromAnnotations(dataset)
print(coords_file)
config['DATA']['N_Classes'] = len(coords_file[config['DATA']['Label']].unique())
del preprocessor

# todo: export current coords_file to numpy. One can then skip preprocessing by using LoadFileParameter.

########################################################################################################################
# 3. Model training

name = GetInfo.format_model_name(config)

# Set up all logging (if training)
if 'logger_folder' in config['CHECKPOINT']:
    logger = TensorBoardLogger(os.path.join('lightning_logs', config['CHECKPOINT']['logger_folder']), name=name)
else:
    logger = TensorBoardLogger('lightning_logs', name=name)
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath=config['CHECKPOINT']['Model_Save_Path'],
                                      monitor=config['CHECKPOINT']['Monitor'],
                                      filename=name + '-epoch{epoch:02d}-' + config['CHECKPOINT']['Monitor'] + '{' +
                                      config['CHECKPOINT']['Monitor'] + ':.2f}',
                                      save_top_k=1,
                                      mode=config['CHECKPOINT']['Mode'])

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

# Load coords_file
# coords_file = LoadFileParameter(config, dataset)

# Augment data on the training set
if config['AUGMENTATION']['Rand_Operations'] > 0:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
            'NORMALIZATION'] else None,
        transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=config['AUGMENTATION']['Rand_Operations'],
                               magnitude=config['AUGMENTATION']['Rand_Magnitude']),
        # this only operates on 8-bit images (not normalised float32 tensors)
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
            'NORMALIZATION'] else None,
        transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# No data augmentation on the validation set
val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data = DataModule(
    coords_file,
    batch_size=config['BASEMODEL']['Batch_Size'],
    train_transform=train_transform,
    val_transform=val_transform,
    train_size=config['DATA']['Train_Size'],
    val_size=config['DATA']['Val_Size'],
    inference=False,
    dim_list=config['BASEMODEL']['Patch_Size'],
    vis_list=config['BASEMODEL']['Vis'],
    n_per_sample=config['DATA']['N_Per_Sample'],
    target=config['DATA']['Label_Name'],
    sampling_scheme=config['DATA']['Sampling_Scheme']
)
config['DATA']['N_Training_Examples'] = data.train_data.__len__()

config['INTERNAL']['weights'] = torch.ones(int(config['DATA']['N_Classes'])).float()
# Return some stats/information on the training/validation data (to explore the dataset / sanity check)
# From paper: Class-balanced Loss Based on Effective Number of Samples

# The following will be used in an upcoming release to add weights to labels. This will be packaged in a function:
# N = sum(npatches_per_class)
# beta = (N-1)/N
# effective_samples = (1 - beta**npatches_per_class)/(1-beta)
# raw_scores = 1 / effective_samples
# w = config['DATA']['N_Classes'] * raw_scores / sum(raw_scores)
# config['INTERNAL']['weights'] = torch.tensor(w).float()
# print(config['INTERNAL']['weights'])
# note: all the above could be moved directly into the ConvNet model.

# Load model and train/infer
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                     precision=config['BASEMODEL']['Precision'], callbacks=[checkpoint_callback, lr_monitor],
                     logger=logger)

model = ConvNet(config)
trainer.fit(model, data)
