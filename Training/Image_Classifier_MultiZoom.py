import numpy as np
import sys
from torchvision import transforms, models
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from Model.ImageClassifier import ImageClassifier
from collections import Counter
from pytorch_lightning.callbacks import ModelSummary,DeviceStatsMonitor

MasterSheet  = sys.argv[1]
SVS_Folder   = sys.argv[2]
Patch_Folder = sys.argv[3]

pl.seed_everything(42)

# Select two WSI manually:
ids = WSIQuery(MasterSheet)

coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)



transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
])  

data = DataModule(coords_file, train_transform=transform, val_transform=transform, batch_size=64, vis_level=0, inference=False, dim=(256, 256), target='tumour_label')
data = DataModule(coords_file, train_transform=transform, val_transform=transform, batch_size=64, vis_level=4, inference=False, dim=(256, 256), target='tumour_label')

model_dict = nn.ModuleDict({
    "low_zoom": ImageClassifier(backbone=models.densenet121(pretrained=False)).backbone
    "high_zoom":  ImageClassifier(backbone=models.densenet121(pretrained=False)).backbone
})

MixModel = MixModel(model_dict)
trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=3,precision=16, callbacks = [ModelSummary()])
res = trainer.fit(model, data)

"""
# Sample code for exporting predicted probabilities.
#dataset    = DataLoader(DataGenerator(coords_file, wsi_file, transform = transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
predictions = trainer.predict(model, dataset)
predicted_sarcoma_classes_probs = np.concatenate(predictions, axis=0)

for i in range(predicted_sarcoma_classes_probs.shape[1]):
    SaveFileParameter(coords_file, Patch_Folder, predicted_sarcoma_classes_probs[:, i], 'sarcoma_pred_label_' + str(i))

# Sample code for future inference
# model = SarcomaClassifier()
# model = torch.load(save_model_path)
# model.eval()
# Verify if load on GPU/CPU is required - https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
