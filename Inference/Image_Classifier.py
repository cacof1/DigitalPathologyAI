from PreProcessing.PreProcessingTools import PreProcessor
import toml
import pytorch_lightning as pl
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet
from Dataloader.Dataloader import *
from Utils import MultiGPUTools

n_gpus = torch.cuda.device_count()  # could go into config file
config = toml.load(sys.argv[1])

########################################################################################################################
# 1. Download all relevant files based on the configuration file

SVS_dataset = QueryFromServer(config)
SynchronizeSVS(config, SVS_dataset)
DownloadNPY(config, SVS_dataset)
print(SVS_dataset)

########################################################################################################################
# 2. Pre-processing: create npy files

# Load pre-processed dataset. It should have been pre-processed with Inference/Preprocess.py first.
tile_dataset = LoadFileParameter(config, SVS_dataset)

# Mask the tile_dataset to only keep the tumour tiles, depending on a pre-set criteria.
tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_tumour'] > 0.85]

########################################################################################################################
# 3. Model + dataloader

# Pad tile_dataset such that the final batch size can be divided by n_gpus.
n_pad = MultiGPUTools.pad_size(len(tile_dataset), n_gpus, config['BASEMODEL']['Batch_Size'])
tile_dataset = MultiGPUTools.pad_dataframe(tile_dataset, n_pad)

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = DataLoader(DataGenerator(tile_dataset, transform=val_transform, svs_folder=config['DATA']['SVS_Folder'], inference=True),
                  batch_size=config['BASEMODEL']['Batch_Size'],
                  num_workers=10,
                  persistent_workers=True,
                  shuffle=False,
                  pin_memory=True)

trainer = pl.Trainer(gpus=n_gpus, strategy='ddp', benchmark=True, precision=config['BASEMODEL']['Precision'],
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

mesenchymal_tumour_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))

for tumour_no, tumour_name in enumerate(mesenchymal_tumour_names):
    tile_dataset['prob_' + config['DATA']['Label'] + '_' + tumour_name] = pd.Series(predicted_classes_prob[:, tumour_no],
                                                                                   index=tile_dataset.index)
    tile_dataset = tile_dataset.fillna(0)

for SVS_ID, df_split in tile_dataset.groupby(tile_dataset.SVS_ID):
    SaveFileParameter(config, df_split, SVS_ID)
