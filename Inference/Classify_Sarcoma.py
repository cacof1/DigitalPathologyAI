# -*- coding: utf-8 -*-

from torchvision import datasets, models, transforms
import os
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataModule, DataGenerator, WSIQuery
from Model.ImageClassifier import ImageClassifier
from __local.SarcomaClassification.Methods import AppendSarcomaLabel
from torchmetrics.functional import accuracy, confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint

# Use same seed for now:
pl.seed_everything(42, workers=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------------------------------------------------
# Generate data

# Master loader
#MasterSheet      = sys.argv[1]
#SVS_Folder       = sys.argv[2]
#Patch_Folder     = sys.argv[3]
#Pretrained_Model = sys.argv[4]

# Local tests - please leave commented
MasterSheet = '../__local/SarcomaClassification/data/NinjaMasterSheet.csv' # sarcoma_diagnoses.csv'
Patch_Folder = '../patches/'
# Pretrained_Model = '../PretrainedModel/sarcoma_classifier/SFTl_SFTh/train_on_5/sarcoma-classifier-epoch03-val_loss_epoch0.01.ckpt'
# Pretrained_Model = '../PretrainedModel/sarcoma_classifier/SFTl_SFTh_DMF_NF/sarcoma-classifier-epoch14-val_loss_epoch0.03.ckpt'
# Pretrained_Model = '../PretrainedModel/sarcoma_classifier/SFTl_SFTh_DMF_NF_SF/sarcoma-classifier-epoch07-val_loss_epoch0.02.ckpt'
Pretrained_Model = '../PretrainedModel/sarcoma_classifier/SFTl_SFTh/sarcoma-classifier-epoch05-val_loss_epoch0.01.ckpt'
Pretrained_Model = '../PretrainedModel/sarcoma_classifier/SFTl_DF/sarcoma-classifier-epoch01-val_loss_epoch0.35.ckpt'

# Classify 5 specific high grade SFT and 5 specific low grade SFTs.
# SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_10samples_SFT_low_high/'
# SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_15samples_SFT_low_high_and_DMF/'
# SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_20samples_SFT_low_high_DMF_and_NF/'
# SVS_Folder = '/home/mikael/Documents/data/digpath/sarcoma_classify_25samples_SFT_low_high_DMF_NF_SF/'
# ids = WSIQuery(MasterSheet, id='484781')  # high grade SFT, label #1
# ids.extend(WSIQuery(MasterSheet, id='484849'))  # high grade SFT, label #1
# ids.extend(WSIQuery(MasterSheet, id='484906'))  # high grade SFT, label #1
# ids.extend(WSIQuery(MasterSheet, id='484978'))  # high grade SFT, label #1
# ids.extend(WSIQuery(MasterSheet, id='485302'))  # high grade SFT, label #1
# ids.extend(WSIQuery(MasterSheet, id='484765'))  # low grade SFT, label #0
# ids.extend(WSIQuery(MasterSheet, id='484808'))  # low grade SFT, label #0
# ids.extend(WSIQuery(MasterSheet, id='484927'))  # low grade SFT, label #0
# ids.extend(WSIQuery(MasterSheet, id='485308'))  # low grade SFT, label #0
# ids.extend(WSIQuery(MasterSheet, id='485336'))  # low grade SFT, label #0
# ids.extend(WSIQuery(MasterSheet, id='492008'))  # Desmoid fibromatosis, label #2
# ids.extend(WSIQuery(MasterSheet, id='492022'))  # Desmoid fibromatosis, label #2
# ids.extend(WSIQuery(MasterSheet, id='492024'))  # Desmoid fibromatosis, label #2
# ids.extend(WSIQuery(MasterSheet, id='492047'))  # Desmoid fibromatosis, label #2
# ids.extend(WSIQuery(MasterSheet, id='492036'))  # Desmoid fibromatosis, label #2
# ids.extend(WSIQuery(MasterSheet, id='492207'))  # Nodular fasciitis, label #3
# ids.extend(WSIQuery(MasterSheet, id='492257'))  # Nodular fasciitis, label #3
# ids.extend(WSIQuery(MasterSheet, id='492274'))  # Nodular fasciitis, label #3
# ids.extend(WSIQuery(MasterSheet, id='492326'))  # Nodular fasciitis, label #3
# ids.extend(WSIQuery(MasterSheet, id='492375'))  # Nodular fasciitis, label #3
# ids.extend(WSIQuery(MasterSheet, id='493081'))  # Superficial fibromatosis, label #4
# ids.extend(WSIQuery(MasterSheet, id='493110'))  # Superficial fibromatosis, label #4
# ids.extend(WSIQuery(MasterSheet, id='493121'))  # Superficial fibromatosis, label #4
# ids.extend(WSIQuery(MasterSheet, id='493127'))  # Superficial fibromatosis, label #4
# ids.extend(WSIQuery(MasterSheet, id='493208'))  # Superficial fibromatosis, label #4
# 492039

SVS_Folder = '/media/mikael/LaCie/sarcoma/'

# Processing #1
# ids = ['486038','485317','484761','484771','484773','484779','484792','484795','484799','484807','484810','484814','484817','484822'] # list of indices for label that were not used for training.
# #       ,'484831','484854','484920','484936','485080','485252','485308','485317','485319','485329','485331','485337','485340']

# Processing #3 - 1st line is DF, 2nd line is SFT low.
ids = [492039,492041,492043,492045,492047,492049,492052,492055,492057,492059,
       484936,485080,485252,485308,485317,485319,485329,485331,485337,485340]


wsi_file, coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)
#coords_file = coords_file[::100]  # To use a subset of the data
coords_file = coords_file[coords_file["tumour_pred_label_1"] > coords_file["tumour_pred_label_0"]]  # only keep the patches labeled as tumour


# Make sure to append sarcoma label too
AppendSarcomaLabel(ids, SVS_Folder, Patch_Folder, mapping_file='mapping_SFTl_DF')

dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform=transform, inference=True), batch_size=50,
                     num_workers=os.cpu_count(), shuffle=False, pin_memory=True)
# should shuffle=False because otherwise it will rotate examples?

# ---------------------------------------------------------------------------------------------------------------------
# # Load model and infer
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=16)
model = ImageClassifier.load_from_checkpoint(Pretrained_Model, backbone=models.densenet121(pretrained=True),
                                             lr=3e-4, num_classes=2)
model.eval()
#

predictions = trainer.predict(model, dataset)
predicted_sarcoma_classes_probs = torch.Tensor.cpu(torch.cat(predictions))
for i in range(predicted_sarcoma_classes_probs.shape[1]):
    SaveFileParameter(coords_file, Patch_Folder, predicted_sarcoma_classes_probs[:, i], 'sarcoma_pred_label_' + str(i))

# ---------------------------------------------------------------------------------------------------------------------
# Export final statistics

# Show final statistics
targets = torch.tensor(dataset.dataset.coords.sarcoma_label.values.astype(int))
# preds = torch.argmax(predicted_sarcoma_classes_probs, dim=1)

tlist = torch.zeros(len(targets),2)
tlist[:,0] = torch.tensor(dataset.dataset.coords.sarcoma_pred_label_0.values)
tlist[:,1] = torch.tensor(dataset.dataset.coords.sarcoma_pred_label_1.values)
preds = torch.argmax(tlist,dim=1)

final_acc = accuracy(preds, targets)
print('Final accuracy over entire dataset is: {}'.format(final_acc))

for pred in preds.unique():
    print('Number of class# {} = : {}/{}'.format(pred,sum(preds == pred), len(preds)))
print('------------------')
# CF = confusion_matrix(preds, targets, model.num_classes)
# print('Confusion matrix:')
# print(CF)
# print('------------------')

# Statistics per SVS.
file_ids = dataset.dataset.coords.file_id.unique()
acc_per_SVS = list()
for file_id in file_ids:
    mask = dataset.dataset.coords.file_id == file_id
    print(sum(mask))
    cur_targets = torch.tensor(dataset.dataset.coords.sarcoma_label.values[mask].astype(int))
    cur_preds = preds[mask.values]
    acc_per_SVS.append(accuracy(cur_preds,cur_targets))