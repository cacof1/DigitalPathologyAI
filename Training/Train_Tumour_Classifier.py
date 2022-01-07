# -*- coding: utf-8 -*-
from torchvision import models, transforms
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from Dataloader.Dataloader import LoadFileParameter, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier
import os
from torch.utils.data import DataLoader
from Dataloader.Dataloader import DataGenerator
from torchmetrics.functional import accuracy, confusion_matrix

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

# Local tests - please leave here
MasterSheet = '../__local/SarcomaClassification/data/NinjaMasterSheet.csv' # sarcoma_diagnoses.csv'  # sys.argv[1]
SVS_Folder = '/home/mikael/Documents/data/digpath/tumor_classify_4samples/'
Patch_Folder = '../patches/'  # sys.argv[3]

# Current working example - with 4 specifically selected svs files.
ids = WSIQuery(MasterSheet, id='484760')
ids.extend(WSIQuery(MasterSheet, id='484761'))
ids.extend(WSIQuery(MasterSheet, id='484763'))
ids.extend(WSIQuery(MasterSheet, id='484764'))

wsi_file, coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)

# coords_file = coords_file[::100]  # Uncomment to train on a subset of the entire dataset

data = DataModule(coords_file, wsi_file, train_transform=transform, val_transform=transform, batch_size=50,
                  inference=False, dim=(256, 256), target='tumour_label', shuffle=True)

# ---------------------------------------------------------------------------------------------------------------------
# Train model

# Save the model with the best monitored property
checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', dirpath='../PretrainedModel/tumour_classifier/',
                                      filename='tumour-classifier-epoch{epoch:02d}-val_loss_epoch{val_loss_epoch:.2f}',
                                      auto_insert_metric_name=False,
                                      mode='min')

trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=50, precision=16,
                     callbacks=[checkpoint_callback])

# Create model
model = ImageClassifier(lr=1e-6, backbone=models.resnet50(pretrained=False))  # create a new one

trainer.fit(model, data)


# ---------------------------------------------------------------------------------------------------------------------
# Sample code for exporting predicted probabilities.

# Re-load data in inference mode
dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform=transform, inference=True), batch_size=50,
                     num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

# Load an inference trainer
inf_trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=16)

# Set model in inference mode
model.eval()

# Predict
predictions = inf_trainer.predict(model, dataset)
predicted_tumour_classes_probs = torch.Tensor.cpu(torch.cat(predictions))

# Show final statistics
targets = torch.tensor(coords_file.tumour_label.values, dtype=torch.int32)
preds = torch.argmax(predicted_tumour_classes_probs, dim=1)
final_acc = accuracy(preds, targets)
print('Final accuracy over entire dataset is: {}'.format(final_acc))
print('Number of class#0 (healthy) : {}/{}'.format(sum(preds == 0), len(preds)))
print('Number of class#1 (tumour)  : {}/{}'.format(sum(preds == 1), len(preds)))
print('------------------')
CF = confusion_matrix(preds, targets, 2)
print('Healthy classified as healthy (TN): {}/{}'.format(CF[0, 0], sum(targets == 0)))
print('Tumour classified as healthy (FP): {}/{}'.format(CF[1, 0], sum(targets == 1)))
print('Healthy classified as tumour (FN): {}/{}'.format(CF[0, 1], sum(targets == 0)))
print('Tumour classified as tumour (TP): {}/{}'.format(CF[1, 1], sum(targets == 1)))
