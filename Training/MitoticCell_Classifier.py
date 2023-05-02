from Dataloader.Dataloader import *
from Dataloader.ObjectDetection import *
import toml
import sys
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torchvision import transforms
import torch
from QA.Normalization.Colour import ColourNorm_old
from Model.MitoticConvNet import ConvNet
from sklearn import preprocessing
from Utils.sampling_schemes import *
import os

config = toml.load('/home/dgs2/Software/DigitalPathologyAI/Configs/config_mitotic_classification.ini')#toml.load(sys.argv[1])
n_gpus = len(config['BASEMODEL']['GPU_ID'])#torch.cuda.device_count()
name = 'MitoticClassification'

#SVS_dataset = QueryFromServer(config)
#SynchronizeSVS(config, SVS_dataset)
#print(SVS_dataset)

logger = TensorBoardLogger(config['CHECKPOINT']['log_path'], name=config['CHECKPOINT']['logger_folder'])

lr_monitor = LearningRateMonitor(logging_interval='step')

checkpoint_callback = ModelCheckpoint(dirpath=config['CHECKPOINT']['Model_Save_Path'],
                                      monitor=config['CHECKPOINT']['Monitor'],
                                      filename=name+'{epoch:02d}-{val_loss_epoch:.4f}_{val_acc_epoch:.4f}',
                                      save_top_k=3,
                                      mode=config['CHECKPOINT']['Mode'])

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

if config['AUGMENTATION']['Rand_Operations'] > 0:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        #ColourNorm_old.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=config['AUGMENTATION']['Rand_Operations'],
                               magnitude=config['AUGMENTATION']['Rand_Magnitude']),
        # this only operates on 8-bit images (not normalised float32 tensors)
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(config['DATA']['dim']),

    ])

else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        #ColourNorm_old.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomHorizontalFlip(config['AUGMENTATION']['Horizontal_Flip']),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(config['DATA']['dim']),

    ])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    #ColourNorm_old.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize(config['DATA']['dim']),

])

# Create LabelEncoder

# Load model and train
trainer = pl.Trainer(accelerator="gpu",
                     devices=config['BASEMODEL']['GPU_ID'],
                     #strategy='ddp_find_unused_parameters_false',
                     benchmark=True,
                     max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                     #precision=config['BASEMODEL']['Precision'],
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=logger,
                     log_every_n_steps=5)

df = pd.read_csv(config['DATA']['Dataframe'])
df_error = pd.read_csv(config['DATA']['Errors_df'])
df = df[~df['nrrd_file'].isin(df_error['nrrd_file'])]
df = df.astype({'SVS_ID':'str'})

df = df[(df['quality'] == 1) & (df['refine'] == 0)]
df = df[df['ann_label'] != '?']

le = preprocessing.LabelEncoder()
le.fit(df['ann_label'])
df['ann_label'] = le.transform(df['ann_label'])
print(list(le.classes_))
config['DATA']['N_Classes'] = len(df['ann_label'].unique())
df.reset_index(drop=True, inplace=True)
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

data = MFDataModule(df_train,
                    df_val,
                    df_test,
                    DataType='MixDataset',
                    masked_input=config['DATA']['masked_input'],
                    #wsi_folder=config['DATA']['SVS_Folder'],
                    #mask_folder=config['DATA']['mask_folder'],
                    nrrd_path = config['DATA']['nrrd_path'],
                    data_source = config['DATA']['data_source'],
                    dim = config['DATA']['dim'],
                    batch_size=config['BASEMODEL']['Batch_Size'],
                    num_of_worker=config['BASEMODEL']['Num_of_Worker'],
                    train_transform=train_transform,
                    val_transform=val_transform,
                    extract_feature=False,
                    inference=False,)

img, label = data.val_data[0]

model = ConvNet(config, label_encoder=le)

trainer.fit(model, data)
#%%
#model = ConvNet(config, label_encoder=le).load_from_checkpoint(checkpoint_path=config['CHECKPOINT']['Model_Save_Path']+"MitoticClassificationepoch=34-val_loss_epoch=0.0803_val_acc_epoch=0.9991.ckpt",)
#trainer = pl.Trainer(accelerator="gpu",devices=[0],benchmark=True)

#trainer.validate(model,dataloaders=data.test_dataloader())
