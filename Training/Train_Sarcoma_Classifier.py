import sys
from torchvision import transforms, models
import torch
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from __local.SarcomaClassification.Methods import AppendSarcomaLabel
from utils import GetInfo
import numpy as np

pl.seed_everything(42, workers=True)

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
SVS_Folder = '/media/mikael/LaCie/sarcoma/'
ids_DF = [492006,492007,492008,492010,492011,492012,492013,492014,492015,492016,492017,492018,492019,492020,492021,492022,492023,492024,492025,492026,492027,492028,492029,492030,492031,
          492033,492034,492035,492036,492037,492038,492040,492042,492044,492046,492048,492051,492054,492056,492058]
ids_SFTl = [484759,484761,484765,484771,484772,484773,484775,484779,484785,484792,484793,484795,484797,484799,484800,484807,484808,484810,484813,484814,484816,484817,484819,484822,484827,
            484831,484847,484854,484859,484920,484927,485078,485089,485254,485315,485318,485328,485330,485336,485338]
ids = list()
for cur_id in ids_SFTl:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_DF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))

# For all files, make sure your target is existing.
AppendSarcomaLabel(ids, SVS_Folder, Patch_Folder, mapping_file='mapping_SFTl_DF')

# Select subset of all data
coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)#, fractional_data=0.05)  # use .5% of data

# Select a subset of coords files
coords_file = coords_file[coords_file["tumour_pred_label_1"] > coords_file["tumour_pred_label_0"]]  # only keep the patches labeled as tumour.

data_fraction = 0.01  # preserve approximately this amount of data
n_per_sample = int(data_fraction * len(coords_file)/len(np.unique(coords_file['file_id'])))
data = DataModule(coords_file, train_transform=transform, val_transform=transform, batch_size=32,
                  inference=False, dim=(256, 256), target='sarcoma_label', n_per_sample=n_per_sample,
                  train_size=0.7, val_size=0.25)  # data.train_data, data.val_data

# ---------------------------------------------------------------------------------------------------------------------
# Train model

# Note: make sure that sarcoma_label column in .csv file has been created prior to running the algorithm:

# Return some stats on what you are doing (for development purposes, knowing your data, etc)
GetInfo.ShowTrainValTestInfo(data)

# Save the model with the best monitored property
checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', dirpath=model_save_path,
                                      filename='sarcoma-classifier-epoch{epoch:02d}-val_loss_epoch{val_loss_epoch:.2f}',
                                      auto_insert_metric_name=False,
                                      mode='min')

# Also save the learning rate (to see what happens with the scheduler)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=100, precision=16,
                     callbacks=[checkpoint_callback, lr_monitor])  # TODO: investigate Deterministic=True.

# Two-labels classification:
#model = ImageClassifier(lr=1e-3, backbone=models.densenet121(pretrained=False))

# Multi-label classification:
#model = ImageClassifier(lr=1e-3, backbone=models.resnet18(pretrained=False), num_classes=2) #drop_rate=0.5), num_classes=2)
model = ImageClassifier(lr=1e-3, backbone=models.densenet121(pretrained=False, drop_rate=0.5), num_classes=2)

#lr_finder = trainer.tuner.lr_find(model, train_dataloader=data.train_dataloader(), val_dataloaders=data.val_dataloader())

print(torch.cuda.memory_allocated(torch.device))

trainer.fit(model, data)

## 
# Sample code for exporting predicted probabilities.
#dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform = transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
#predictions = trainer.predict(model, dataset)
#predicted_sarcoma_classes_probs = np.concatenate(predictions, axis=0)

#for i in range(predicted_sarcoma_classes_probs.shape[1]):
#    SaveFileParameter(coords_file, Patch_Folder, predicted_sarcoma_classes_probs[:, i], 'sarcoma_pred_label_' + str(i))