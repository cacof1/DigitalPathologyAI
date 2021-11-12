import numpy as np
import sys
from torchvision import transforms, models
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier

# Example to achieve sarcoma types classification with the ImageClassifier class.

# Option to run with or without arguments. Will be updated with parser in the near future.
if len(sys.argv) == 1:
    MasterSheet = '../__local/SarcomaClassification/data/sarcoma_diagnoses.csv'  # sys.argv[1]
    SVS_Folder = '/Users/mikael/Dropbox/M/PostDoc/UCL/datasets/Digital_Pathology/sft_first_comparison/'
    Patch_Folder = '../patches/'  # sys.argv[3]
else:
    MasterSheet = sys.argv[1]
    SVS_Folder = sys.argv[2]
    Patch_Folder = sys.argv[3]

pl.seed_everything(42)

# Query WSI of interest. Some examples below:

# Select 10 WSI of each SFT low and SFT high for training:
#ids = WSIQuery(MasterSheet, diagnosis='solitary_fibrous_tumour', grade='low')[:10]
#ids.extend(WSIQuery(MasterSheet, diagnosis='solitary_fibrous_tumour', grade='high')[:10])

# Select two WSI manually:
ids = WSIQuery(MasterSheet, id=484757)
ids.extend(WSIQuery(MasterSheet, id=484772))

wsi_file, coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)

# Select a subset of coords files
coords_file = coords_file[coords_file.index < 20]  # keep the first 200 patches of each WSI
#coords_file = coords_file[coords_file["tumour_label"] == 1]  # only keep the patches labeled as tumour

transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  # Required transforms according to resnet/densenet documentation

data = DataModule(coords_file, wsi_file, train_transform=transform, val_transform=transform, batch_size=4,
                    inference=False, dim=(256, 256), target='sarcoma_label')

model = ImageClassifier(backbone=models.densenet121(pretrained=False))
# model = SarcomaClassifier.load_from_checkpoint(sys.argv[2]) # to load from a previous checkpoint

trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=3)
res = trainer.fit(model, data)

# Sample code for exporting predicted probabilities.
dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform = transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
predictions = trainer.predict(model, dataset)
predicted_sarcoma_classes_probs = np.concatenate(predictions, axis=0)

for i in range(predicted_sarcoma_classes_probs.shape[1]):
    SaveFileParameter(coords_file, Patch_Folder, predicted_sarcoma_classes_probs[:, i], 'sarcoma_pred_label_' + str(i))

# Sample code for future inference
# model = SarcomaClassifier()
# model = torch.load(save_model_path)
# model.eval()
# Verify if load on GPU/CPU is required - https://pytorch.org/tutorials/beginner/saving_loading_models.html