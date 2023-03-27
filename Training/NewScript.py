from Dataloader.Dataloader import *
import toml
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torchvision import transforms
import torch
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet
from sklearn import preprocessing
import datetime

n_gpus = torch.cuda.device_count()  # could go into config file
#config = toml.load(sys.argv[1])
# config = toml.load('/home/dgs/Dropbox/M/PostDoc/UCL/Code/Python/DigitalPathologyAI/Configs/preprocessing/infer_tumour_convnet_7classes.ini')
config = toml.load('/Users/mikael/Dropbox/M/PostDoc/UCL/Code/Python/DigitalPathologyAI/Configs/sarcoma/trainer_sarcoma_convnet_15types.ini')

########################################################################################################################
# 1. Download all relevant files based on the configuration file

SVS_dataset = QueryImageFromCriteria(config)
SynchronizeSVS(config, SVS_dataset)
DownloadNPY(config, SVS_dataset)
print(SVS_dataset)