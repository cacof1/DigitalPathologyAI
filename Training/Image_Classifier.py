from torchvision import transforms
import torch
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import toml
from utils import GetInfo

from __local.SarcomaClassification.Methods import AppendSarcomaLabel

# Load configuration file and name
config = toml.load('SarcomaTrainer.ini')
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

if config['DATA']['target'] == 'sarcoma_label':  # TODO : potentially move the following step out of Image_Classifier
    # Specific to sarcoma study: make sure that all ids have their "sarcoma_label" target.
    # For another target, make sure you use your own function to append your targets to csv files.
    AppendSarcomaLabel(ids, config['DATA']['SVS_Folder'], config['DATA']['Patches_Folder'], mapping_file='mapping_SFTl_DF_NF_SF')

# Load coords_file
coords_file = LoadFileParameter(ids, config['DATA']['SVS_Folder'], config['DATA']['Patches_Folder'])


if config['DATA']['target'] == 'sarcoma_label': # TODO: maybe encode more efficiently in the config file.
    # Select a subset of coords files. In the sarcoma study, we only consider patches labelled as tumour.
    coords_file = coords_file[coords_file["tumour_pred_label_1"] > coords_file["tumour_pred_label_0"]]  # only keep the patches labeled as tumour.

transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  

data      = DataModule(
    coords_file,
    batch_size       =config['MODEL']['Batch_Size'],
    train_transform  =transform,
    val_transform    =transform,
    inference        =False,
    dim_list         =config['DATA']['dim'],
    vis_list         =config['DATA']['vis'],
    n_per_sample     =config['DATA']['n_per_sample'],
    target           =config['DATA']['target']
)

# Return some stats/information on the processed data (to explore the dataset / sanity check)
if config['VERBOSE']['data_info']:
    GetInfo.ShowTrainValTestInfo(data)

# Load model and classify
model   = ImageClassifier(config)
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=config['MODEL']['Max_Epochs'],
                     precision=config['MODEL']['Precision'], callbacks=[checkpoint_callback], logger=logger)
res     = trainer.fit(model, data)

