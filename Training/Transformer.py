from torchvision import transforms
import torch
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, DataModule, WSIQuery, DataGenerator
from Model.Transformer import ViT
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import toml,sys
from utils import GetInfo
from torch.utils.data import DataLoader

# Load configuration file and name
config = toml.load(sys.argv[1])
#config = toml.load('TransformerTrainerSarcoma.ini')
name = GetInfo.format_model_name(config)

# Set up all logging
logger = TensorBoardLogger('lightning_logs', name=name)
checkpoint_callback = ModelCheckpoint(
    dirpath     =config['MODEL']['Model_Save_Path'],
    monitor     =config['CHECKPOINT']['monitor'],
    filename    =name + '-epoch{epoch:02d}-' + config['CHECKPOINT']['monitor'] + '{' + config['CHECKPOINT']['monitor'] + ':.2f}',
    save_top_k  =1,
    mode        =config['CHECKPOINT']['mode'])

pl.seed_everything(config['MODEL']['RANDOM_SEED'], workers=True)

# Return WSI according to the selected CRITERIA in the configuration file.
ids = WSIQuery(config)

# Load coords_file
coords_file = LoadFileParameter(ids, config['DATA']['SVS_Folder'], config['DATA']['Patches_Folder'])
transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  

if config['MODEL']['inference'] is False:  # train
    data = DataModule(
        coords_file,
        batch_size=config['MODEL']['Batch_Size'],
        train_transform=transform,
        val_transform=transform,
        inference=False,
        dim_list=config['DATA']['dim'],
        vis_list=config['DATA']['vis'],
        n_per_sample=config['DATA']['n_per_sample'],
        target=config['DATA']['target']
    )
else:  # prediction does not use train/validation sets, only directly the dataloader.
    data = DataLoader(DataGenerator(coords_file, transform=transform, inference=True),
                      batch_size=config['MODEL']['Batch_Size'],
                      num_workers=10,
                      shuffle=False,
                      pin_memory=True)


# Load model and classify
model   = ViT(config)
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=config['MODEL']['Max_Epochs'],
                     precision=config['MODEL']['Precision'], callbacks=[checkpoint_callback], logger=logger)
res     = trainer.fit(model, data)

