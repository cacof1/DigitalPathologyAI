import sys
from torchvision import transforms, models
import torch
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from __local.SarcomaClassification.Methods import AppendSarcomaLabel
from utils import GetInfo
from datetime import date
import torch.nn as nn
import numpy as np

# ----------------------------------------------------------------------------------------------------
# Set training parameters (will be updated with the CONFIG file of Charles)
config = dict()
config['backbone'] = 'densenet121'
config['batch_size'] = 32
config['checkpoint_mode'] = 'max'
config['checkpoint_monitor'] = 'val_acc'
config['dim'] = (256, 256)
config['drop_rate'] = 0.1
config['init_seed'] = 12
config['lr'] = 1e-3
config['loss_fcn'] = 'CrossEntropyLoss'
config['max_epochs'] = 100
config['n_per_sample'] = 500
config['num_classes'] = 4
config['precision'] = 16
config['pretrained'] = True
config['target'] = 'sarcoma_label'
config['train_size'] = 0.7
config['val_size'] = 0.29

extra = ''
model_name = config['backbone'] + '_pre' if config['pretrained'] is True else config['backbone']
model_name = model_name + 'drop' + str(config['drop_rate']) if 'densenet' in model_name else model_name

config['name'] = model_name + '_lr' + str(config['lr']) + '_dim' + str(config['dim'][0]) + '_batch' + \
                 str(config['batch_size']) + '_N' + str(config['num_classes']) + '_n' + str(config['n_per_sample'])\
                 + '_epochs' + str(config['max_epochs']) + '_train' + str(int(100*config['train_size'])) + '_val'\
                 + str(int(100*config['val_size'])) + '_loss' + config['loss_fcn'] + '_' \
                 + '_seed' + str(config['init_seed']) + date.today().strftime("%b-%d") + extra
print('File name will be: {}'.format(config['name']))
# ----------------------------------------------------------------------------------------------------

pl.seed_everything(config['init_seed'], workers=True)

# Example to achieve sarcoma types classification with the ImageClassifier class.

# Option to run with or without arguments. TODO: update with parser in the near future.
if len(sys.argv) == 1:
    MasterSheet = '../__local/SarcomaClassification/data/NinjaMasterSheet.csv' # sarcoma_diagnoses.csv'  # sys.argv[1]
    #print('gotta start using my own file again, but convert numbers to strings!')
    #SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_10samples_SFT_low_high/'
    # SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_15samples_SFT_low_high_and_DMF/'
    # SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_20samples_SFT_low_high_DMF_and_NF/'
    # SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_25samples_SFT_low_high_DMF_NF_SF/'  # contains all files !
    Patch_Folder = '../patches/'  # sys.argv[3]
    model_save_path = '../PretrainedModel/sarcoma_classifier/SFTl_DF/with_dropout'
    # model_save_path = '../PretrainedModel/sarcoma_classifier/SFTl_SFTh_DMF_NF/'
    # model_save_path = '../PretrainedModel/sarcoma_classifier/SFTl_SFTh_DMF_NF_SF/'
    model_save_path = '../PretrainedModel/sarcoma_classifier/SFTl_DF_NF_SF/'
else:
    MasterSheet = sys.argv[1]
    SVS_Folder = sys.argv[2]
    Patch_Folder = sys.argv[3]
    model_save_path = sys.argv[4]

transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  # Required transforms according to resnet/densenet documentation

# ---------------------------------------------------------------------------------------------------------------------
# Generate data

# Select 10 WSI of each SFT low and SFT high for training:
#ids = WSIQuery(MasterSheet, diagnosis='solitary_fibrous_tumour', grade='low')[:10]
#ids.extend(WSIQuery(MasterSheet, diagnosis='solitary_fibrous_tumour', grade='high')[:10])

# Processing #1: on 25 patients for each slice.
# SVS_Folder = '/media/mikael/LaCie/sarcoma/'
# ids_SFTh = [484757,484781,484849,484906,484945,484959,484978,485045,485074,485077,485081,485099,485109,485121,485250,485295,485302,485303,485333,485349,485395,485435,485491,485512,485515]
# ids_SFTl = [484759,484765,484772,484775,484785,484793,484797,484800,484808,484813,484816,484819,484827,484847,484859,484927,485078,485089,485254,485315,485318,485328,485330,485336,485338]
# ids = list()
# for cur_id in ids_SFTl:
#     ids.extend(WSIQuery(MasterSheet,id = str(cur_id)))
# for cur_id in ids_SFTh:
#     ids.extend(WSIQuery(MasterSheet,id = str(cur_id)))

# Processing #3: on 40 patients of each slice (SFTl vs DF).
# SVS_Folder = '/media/mikael/LaCie/sarcoma/'
# ids_DF = [492006,492007,492008,492010,492011,492012,492013,492014,492015,492016,492017,492018,492019,492020,492021,492022,492023,492024,492025,492026,492027,492028,492029,492030,492031,
#           492033,492034,492035,492036,492037,492038,492040,492042,492044,492046,492048,492051,492054,492056,492058]
# ids_SFTl = [484759,484761,484765,484771,484772,484773,484775,484779,484785,484792,484793,484795,484797,484799,484800,484807,484808,484810,484813,484814,484816,484817,484819,484822,484827,
#             484831,484847,484854,484859,484920,484927,485078,485089,485254,485315,485318,485328,485330,485336,485338]
# ids = list()
# for cur_id in ids_SFTl:
#     ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
# for cur_id in ids_DF:
#     ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
# AppendSarcomaLabel(ids, SVS_Folder, Patch_Folder, mapping_file='mapping_SFTl_DF')

# Processing #4: on 160 patients total (40 slices per case, SFTL, DF, NF, SF.)
SVS_Folder = '/media/mikael/LaCie/sarcoma/'
ids_DF = [492006,492007,492008,492010,492011,492012,492013,492014,492015,492016,492017,492018,492019,492020,492021,492022,492023,492024,492025,492026,492027,492028,492029,492030,492031,
            492033,492034,492035,492036,492037,492038,492040,492042,492044,492046,492048,492051,492054,492056,492058]
ids_SFTl = [484759,484761,484765,484771,484772,484773,484775,484779,484785,484792,484793,484795,484797,484799,484800,484807,484808,484810,484813,484814,484816,484817,484819,484822,484827,
            484831,484847,484854,484859,484920,484927,485078,485089,485254,485315,485318,485328,485330,485336,485338]
ids_NF = [492207,492208,492209,492234,492257,492259,492265,492266,492267,492269,492270,492271,492272,492273,492274,492276,492277,492280,492281,492283,
            492295,492363,492311,492326,492329,492333,492339,492343,492352,492356,492358,492372,492381,492392,492408,492418,492436,492447,492451,492461]
ids_SF = [493065,493208,493068,493157,493071,493072,493073,493074,493075,493076,493077,493079,493080,493081,493082,493084,493088,493093,
            493096,493103,493186,493199,493110,493113,493116,493118,493119,493121,493122,493127,493131,493148,493168,493197,493203,493211,493220,493226,493239,493250]

ids = list()
for cur_id in ids_SFTl:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_DF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_NF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_SF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))

# For all files, make sure your target is existing.
AppendSarcomaLabel(ids, SVS_Folder, Patch_Folder, mapping_file='mapping_SFTl_DF_NF_SF')

# Select subset of all data
coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)

# Select a subset of coords files
coords_file = coords_file[coords_file["tumour_pred_label_1"] > coords_file["tumour_pred_label_0"]]  # only keep the patches labeled as tumour.

#data_fraction = 0.04  # preserve approximately this amount of data
#n_per_sample = int(data_fraction * len(coords_file)/len(np.unique(coords_file['file_id'])))  # attention, on revoit le mm example...
#n_per_sample = 500  # verifier pk 1633 marche pas
data = DataModule(coords_file, train_transform=transform, val_transform=transform, batch_size=config['batch_size'],
                  inference=False, dim=config['dim'], target=config['target'], n_per_sample=config['n_per_sample'],
                  train_size=config['train_size'], val_size=config['val_size'])  # data.train_data, data.val_data

# ---------------------------------------------------------------------------------------------------------------------
# Train model

# Note: make sure that sarcoma_label column in .csv file has been created prior to running the algorithm:

# Return some stats on what you are doing (for development purposes, knowing your data, etc)
GetInfo.ShowTrainValTestInfo(data)

# Save the model with the best monitored property
checkpoint_callback = ModelCheckpoint(monitor=config['checkpoint_monitor'], dirpath=model_save_path,
                                      filename='sarcoma-classifier-epoch{epoch:02d}-val_loss_epoch{val_loss_epoch:.2f}',
                                      auto_insert_metric_name=False,
                                      mode=config['checkpoint_mode'])

# Also save the learning rate (to see what happens with the scheduler)
lr_monitor = LearningRateMonitor(logging_interval='step')

# Set logger name
logger = TensorBoardLogger('lightning_logs', name=config['name'])

trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=config['max_epochs'], precision=config['precision'],
                     callbacks=[checkpoint_callback, lr_monitor], logger=logger)  # TODO: investigate Deterministic=True.

# Two-labels classification:
#model = ImageClassifier(lr=1e-3, backbone=models.densenet121(pretrained=False))

# Multi-label classification:
backbone = getattr(models, config['backbone'])
if 'densenet' in config['backbone']:
    backbone = backbone(pretrained=config['pretrained'], drop_rate=config['drop_rate'])
else:
    backbone = backbone(pretrained=config['pretrained'])

lossfcn = getattr(nn, config['loss_fcn'])
model = ImageClassifier(lr=config['lr'], backbone=backbone, num_classes=config['num_classes'], lossfcn=lossfcn())
#model = ImageClassifier(lr=1e-3, backbone=models.densenet121(pretrained=True, drop_rate=0.1), num_classes=2)
#model = ImageClassifier(lr=1e-3, backbone=models.densenet121(pretrained=False, drop_rate=0.1), num_classes=4)
#model = ImageClassifier(lr=1e-3, backbone=models.resnet34(pretrained=True), num_classes=4) #drop_rate=0.5), num_classes=2)
#lr_finder = trainer.tuner.lr_find(model, train_dataloader=data.train_dataloader(), val_dataloaders=data.val_dataloader())

trainer.fit(model, data)

## 
# Sample code for exporting predicted probabilities.
#dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform = transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
#predictions = trainer.predict(model, dataset)
#predicted_sarcoma_classes_probs = np.concatenate(predictions, axis=0)

#for i in range(predicted_sarcoma_classes_probs.shape[1]):
#    SaveFileParameter(coords_file, Patch_Folder, predicted_sarcoma_classes_probs[:, i], 'sarcoma_pred_label_' + str(i))