import toml
import pytorch_lightning as pl
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet
from Utils import MultiGPUTools
import multiprocessing as mp
from pathlib import Path
from Dataloader.Dataloader import *

n_gpus = torch.cuda.device_count()  # could go into config file
config = toml.load(sys.argv[1])

########################################################################################################################
# 1. Download all relevant files based on the configuration file

SVS_dataset = QueryImageFromCriteria(config)
SynchronizeSVS(config, SVS_dataset)
DownloadNPY(config, SVS_dataset)
print(SVS_dataset)

########################################################################################################################
# 2. Pre-processing: create npy files

# Load pre-processed dataset. It should have been pre-processed (tissue type identification) first.
print('Loading file parameters...', end='')
tile_dataset = LoadFileParameter(config, SVS_dataset)
tile_dataset_full = tile_dataset.copy()  # keep the full tile_dataset for final saving, but only process the reduced.
valid_tumour_tiles_index = tile_dataset_full['prob_tissue_type_Tumour'] > 0.94
tile_dataset = tile_dataset.loc[valid_tumour_tiles_index]

# Assign SVS paths based on SVS_dataset (this will change depending on which workstation is running the script).
tile_dataset['SVS_PATH'] = tile_dataset['id_external'].map(dict(zip(SVS_dataset.id_external, SVS_dataset.SVS_PATH)))
print('Done.')

########################################################################################################################
# 3. Model + dataloader

# Pad tile_dataset such that the final batch size can be divided by n_gpus.
n_pad = MultiGPUTools.pad_size(len(tile_dataset), n_gpus, config['BASEMODEL']['Batch_Size'])
tile_dataset = MultiGPUTools.pad_dataframe(tile_dataset, n_pad)

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = DataLoader(DataGenerator(tile_dataset, transform=val_transform, target=config['DATA']['Label'], inference=True),
                  batch_size=config['BASEMODEL']['Batch_Size'],
                  num_workers=int(.8 * mp.Pool()._processes),
                  persistent_workers=True,
                  shuffle=False,
                  pin_memory=True)

trainer = pl.Trainer(gpus=n_gpus,
                     strategy='ddp',
                     benchmark=False,
                     precision=config['BASEMODEL']['Precision'],
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
model.eval()

########################################################################################################################
# 4. Predict

predictions = trainer.predict(model, data)
ordered_preds = MultiGPUTools.reorder_predictions(predictions)  # reorder if processing was done on multiple GPUs
predicted_classes_prob = torch.Tensor.cpu(torch.cat(ordered_preds))

# Drop padding
tile_dataset = tile_dataset.iloc[:-n_pad]
predicted_classes_prob = predicted_classes_prob[:-n_pad]

########################################################################################################################
# 5. Save locally (no upload to OMERO for the sarcoma classification yet)

print('Saving sarcoma classification results locally to npy files...')

mesenchymal_tumour_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))

# Append tumour type probabilities to tumour tiles.
for tumour_no, tumour_name in enumerate(mesenchymal_tumour_names):
    curkey = 'prob_' + config['DATA']['Label'] + '_' + tumour_name
    tile_dataset_full[curkey] = np.nan
    tile_dataset_full.loc[valid_tumour_tiles_index, curkey] = pd.Series(predicted_classes_prob[:, tumour_no], index=tile_dataset.index)

for id_external, df_split in tile_dataset_full.groupby(tile_dataset_full.id_external):
    npy_path = SaveFileParameter(config, df_split, str(id_external))
    print('File exported at {}.'.format(npy_path))

print('Done.')

###############################################################################################