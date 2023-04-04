from Dataloader.Dataloader import *
import toml
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torchvision import transforms
import torch
from QA.Normalization.Colour import ColourNorm, ColourAugment
from Model.ConvNet import ConvNet
from sklearn import preprocessing
import datetime

n_gpus = torch.cuda.device_count()  # could go into config file
config = toml.load(sys.argv[1])
print("{} GPUs are used for training".format(n_gpus))

########################################################################################################################
# 1. Download all relevant files based on the configuration file

SVS_dataset = QueryImageFromCriteria(config)
SynchronizeSVS(config, SVS_dataset)
DownloadNPY(config, SVS_dataset)
#SVS_dataset = SVS_dataset.groupby('diagnosis').head(n=5) # uncomment for quick testing.
print(SVS_dataset)

########################################################################################################################
# 2. Pre-processing: load existing npy files, append target label to tile_dataset, select tumour tiles only.

# Load pre-processed dataset. It should have been pre-processed with Inference/Preprocess.py first.
print('Loading file parameters...', end='')
tile_dataset = LoadFileParameter(config, SVS_dataset)
tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_Tumour'] > 0.94]  # keep only tumour tiles
print('Done.')

# Append the target label to tile_dataset.
print('Appending target label to tile_dataset...',end='')
tile_dataset[config['DATA']['Label']] = tile_dataset['id_external'].map(dict(zip(SVS_dataset.id_external, SVS_dataset.diagnosis)))
print('Done.')

# Print number of training classes and the number of examples for each.
config['DATA']['N_Classes'] = len(tile_dataset[config['DATA']['Label']].unique())
print('There are {} classes in the training dataset.'.format(config['DATA']['N_Classes']))
print(tile_dataset.value_counts(subset=config['DATA']['Label']))
########################################################################################################################
# 3. Model

# Set up logging, model checkpoint
name = GetInfo.format_model_name(config)
if 'logger_folder' in config['CHECKPOINT']:
    logger = TensorBoardLogger(os.path.join('lightning_logs', config['CHECKPOINT']['logger_folder']), name=name)
else:
    logger = TensorBoardLogger('lightning_logs', name=name)

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir,
                                      monitor=config['CHECKPOINT']['Monitor'],
                                      filename=name + '-epoch{epoch:02d}-' + config['CHECKPOINT']['Monitor'] + '{' +
                                               config['CHECKPOINT']['Monitor'] + ':.2f}',
                                      save_top_k=1,
                                      mode=config['CHECKPOINT']['Mode'])

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

# transforms: augment data on training set
train_transform = transforms.Compose([
    transforms.ToTensor(),  # Normalises to [0, 1]
    ColourAugment.ColourAugment(sigma=config['AUGMENTATION']['Colour_Sigma'], mode=config['AUGMENTATION']['Colour_Mode']),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.ToTensor(),  # this also normalizes to [0,1].,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# transforms: colour norm only on validation set
val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create LabelEncoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(tile_dataset[config['DATA']['Label']])

# Load model and train
trainer = pl.Trainer(gpus=n_gpus,
                     strategy='ddp',
                     benchmark=False,
                     max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                     precision=config['BASEMODEL']['Precision'],
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=logger)

model = ConvNet(config, label_encoder=label_encoder)
########################################################################################################################
# 4. Dataloader

data = DataModule(
    tile_dataset,
    batch_size=config['BASEMODEL']['Batch_Size'],
    train_transform=train_transform,
    val_transform=val_transform,
    train_size=config['DATA']['Train_Size'],
    val_size=config['DATA']['Val_Size'],
    test_size=config['DATA']['Test_Size'],
    inference=False,
    dim=config['BASEMODEL']['Patch_Size'],
    vis=config['BASEMODEL']['Vis'][0],
    n_per_sample=config['DATA']['N_Per_Sample'],
    target=config['DATA']['Label'],
    label_encoder=label_encoder
)
# Give the user some insight on the data
GetInfo.ShowTrainValTestInfo(data, config)

# Set default parameters for loss weights (currently unused)
#config['DATA']['N_Training_Examples'] = data.train_data.__len__()
#config['DATA']['loss_weights'] = torch.ones(int(config['DATA']['N_Classes'])).float()

# Train/validate
trainer.fit(model, data)

# Test
trainer.test(model, data.test_dataloader())

# Write config file in logging folder for safekeeping
with open(logger.log_dir + "/Config.ini", "w+") as toml_file:
    toml.dump(config, toml_file)
    toml_file.write("Train transform:\n")
    toml_file.write(str(train_transform))
    toml_file.write("Val/Test transform:\n")
    toml_file.write(str(val_transform))
