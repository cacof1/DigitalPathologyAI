# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:33:52 2021
@author: zhuoy/cacfek
"""

from torchvision import models
import os
import sys
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataModule, DataGenerator, WSIQuery
from Model.ImageClassifier import ImageClassifier

# Master loader
if len(sys.argv) == 1:
    MasterSheet = '../__local/SarcomaClassification/data/NinjaMasterSheet.csv'
    SVS_Folder = '/media/mikael/LaCie/sarcoma/'
    Patch_Folder = '../patches/'
    Pretrained_Model = '../PretrainedModel/tumour_classifier/tumour-classifier-epoch27-val_loss_epoch0.05.ckpt'
else:
    MasterSheet = sys.argv[1]
    SVS_Folder = sys.argv[2]
    Patch_Folder = sys.argv[3]
    Pretrained_Model = sys.argv[4]

pl.seed_everything(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load curated list of svs.
ids_DF = [492006, 492007, 492008, 492010, 492011, 492012, 492013, 492014, 492015, 492016, 492017, 492018, 492019,
          492020, 492021, 492022, 492023, 492024, 492025, 492026, 492027, 492028, 492029, 492030, 492031,
          492033, 492034, 492035, 492036, 492037, 492038, 492040, 492042, 492044, 492046, 492048, 492051, 492054,
          492056, 492058]
ids_SFTl = [484759, 484761, 484765, 484771, 484772, 484773, 484775, 484779, 484785, 484792, 484793, 484795, 484797,
            484799, 484800, 484807, 484808, 484810, 484813, 484814, 484816, 484817, 484819, 484822, 484827,
            484831, 484847, 484854, 484859, 484920, 484927, 485078, 485089, 485254, 485315, 485318, 485328, 485330,
            485336, 485338]
ids_NF = [492207, 492208, 492209, 492234, 492257, 492259, 492265, 492266, 492267, 492269, 492270, 492271, 492272,
          492273, 492274, 492276, 492277, 492280, 492281, 492283,
          492295, 492363, 492311, 492326, 492329, 492333, 492339, 492343, 492352, 492356, 492358, 492372, 492381,
          492392, 492408, 492418, 492436, 492447, 492451, 492461]
ids_SF = [493065, 493208, 493068, 493157, 493071, 493072, 493073, 493074, 493075, 493076, 493077, 493079, 493080,
          493081, 493082, 493084, 493088, 493093,
          493096, 493103, 493186, 493199, 493110, 493113, 493116, 493118, 493119, 493121, 493122, 493127, 493131,
          493148, 493168, 493197, 493203, 493211, 493220, 493226, 493239, 493250]
ids = list()
for cur_id in ids_SFTl:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_DF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_NF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))
for cur_id in ids_SF:
    ids.extend(WSIQuery(MasterSheet, id=str(cur_id)))

# Load data
coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)
dataset = DataLoader(DataGenerator(coords_file, transform=transform, inference=True), batch_size=50,
                     num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

#data = DataModule(coords_file, train_transform=transform, val_transform=transform, batch_size=32,
#                  inference=False, dim=(256, 256), target='sarcoma_label', n_per_sample=n_per_sample,
#                  train_size=0.7, val_size=0.29)  # data.train_data, data.val_data

# Load existing model and set in inference mode
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=16)
model = ImageClassifier.load_from_checkpoint(Pretrained_Model, backbone=models.resnet50(pretrained=False), lr=1e-6)
model.eval()

# Make predictions and export
predictions = trainer.predict(model, dataset)
predicted_tumour_classes_probs = torch.Tensor.cpu(torch.cat(predictions))
for i in range(predicted_tumour_classes_probs.shape[1]):
    SaveFileParameter(coords_file, Patch_Folder, predicted_tumour_classes_probs[:, i], 'tumour_pred_label_' + str(i))
