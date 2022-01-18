import numpy as np
import sys
from torchvision import transforms, models
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier
from collections import Counter
from pytorch_lightning.callbacks import ModelSummary,DeviceStatsMonitor

config   = toml.load(sys.argv[1])
name     = config['MODEL']['BaseModel'] +"_"+ config['MODEL']['Backbone']+ "_wf" + str(config['MODEL']['wf']) + "_depth" + str(config['MODEL']['depth'])
logger   = TensorBoardLogger('lightning_logs',name = name)
checkpoint_callback = ModelCheckpoint(
    dirpath     = logger.log_dir,
    monitor     = 'val_loss',
    filename    = name,
    save_top_k  = 1,
    mode        = 'min')

MasterSheet    = config['DATA']['Mastersheet']
SVS_Folder     = config['DATA']['SVS_Folder']
Patches_Folder = config['DATA']['Patches_Folder']
seed_everything(config['MODEL']['RANDOM_SEED'])

# Select two WSI manually:
ids = WSIQuery(MasterSheet, config)

coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)

transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  

data      = DataModule(
    coords_file,
    batch_size       = config['MODEL']['Batch_Size'],
    train_transform  = train_transform,
    val_transform    = val_transform,
    inference        = False,
    dim_list         = config['DATA']['dim'],
    vis_list         = config['DATA']['vis'],
    n_per_sample     = config['DATA']['n_per_sample'],
    target           = config['DATA']['target']
)

model   = ImageClassifier(backbone=models.densenet121(pretrained=False))
trainer = pl.Trainer(gpus=1, max_epochs=config['MODEL']['Max_Epochs'],precision=config['MODEL']['Precision'], callbacks = callbacks,logger=logger)
res     = trainer.fit(model, data)

