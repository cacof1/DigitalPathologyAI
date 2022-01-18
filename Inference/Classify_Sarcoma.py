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

# Local tests - please leave commented
MasterSheet = '../__local/SarcomaClassification/data/NinjaMasterSheet.csv' # sarcoma_diagnoses.csv'
Patch_Folder = '../patches/'
Pretrained_Model = '../PretrainedModel/sarcoma_classifier/SFTl_DF/sarcoma-classifier-epoch01-val_loss_epoch0.35.ckpt'
SVS_Folder = '/media/mikael/LaCie/sarcoma/'

# Processing #3 - 1st line is DF, 2nd line is SFT low.
ids = [492039,492041,492043,492045,492047,492049,492052,492055,492057,492059,
       484936,485080,485252,485308,485317,485319,485329,485331,485337,485340]

coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)
coords_file = coords_file[coords_file["tumour_pred_label_1"] > coords_file["tumour_pred_label_0"]]  # only keep the patches labeled as tumour

# Make sure to append sarcoma label too
AppendSarcomaLabel(ids, SVS_Folder, Patch_Folder, mapping_file='mapping_SFTl_DF')

dataset = DataLoader(DataGenerator(coords_file, transform=transform, inference=True), batch_size=50,
                     num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

# ---------------------------------------------------------------------------------------------------------------------
# # Load model and infer
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=16)
model = ImageClassifier.load_from_checkpoint(Pretrained_Model, backbone=models.densenet121(pretrained=True),
                                             lr=3e-4, num_classes=2)
model.eval()

predictions = trainer.predict(model, dataset)
predicted_sarcoma_classes_probs = torch.Tensor.cpu(torch.cat(predictions))
for i in range(predicted_sarcoma_classes_probs.shape[1]):
    SaveFileParameter(coords_file, Patch_Folder, predicted_sarcoma_classes_probs[:, i], 'sarcoma_pred_label_' + str(i))

# ---------------------------------------------------------------------------------------------------------------------
# Export final statistics

# Show final statistics
targets = torch.tensor(dataset.dataset.coords.sarcoma_label.values.astype(int))
# preds = torch.argmax(predicted_sarcoma_classes_probs, dim=1)

tlist = torch.zeros(len(targets), 2)
tlist[:, 0] = torch.tensor(dataset.dataset.coords.sarcoma_pred_label_0.values)
tlist[:, 1] = torch.tensor(dataset.dataset.coords.sarcoma_pred_label_1.values)
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