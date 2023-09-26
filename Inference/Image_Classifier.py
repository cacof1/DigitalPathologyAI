import sys
import os
import datetime
import torch
from torch import cuda
from torchvision import transforms
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
import lightning as L
import toml
from sklearn import preprocessing
from lightning.pytorch.strategies import DDPStrategy
from Dataloader.Dataloader import *
from Utils import GetInfo
from Model.ConvNet import ConvNet
from QA.Normalization.Colour import ColourAugment
import datetime
from Utils import MultiGPUTools
from Utils.OmeroTools import (
    QueryImageFromCriteria,
    SynchronizeSVS,
    SynchronizeNPY
)
def load_config(config_file):
    return toml.load(config_file)

def get_tile_dataset(config):
    SVS_dataset = QueryImageFromCriteria(config)
    SynchronizeSVS(config, SVS_dataset)    
    SynchronizeNPY(config, SVS_dataset)
    tile_dataset = LoadFileParameter(config, SVS_dataset)
    tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_Tumour'] > config['BASEMODEL']['Prob_Tumour_Tresh']]
    tile_dataset = tile_dataset.merge(SVS_dataset, on='id_external')
    tile_dataset['SVS_PATH'] = tile_dataset['SVS_PATH_y'] # Ugly
    return tile_dataset#, tile_dataset_full, valid_tumour_tiles_index

def get_transforms(config):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def Inference(config_file):

    n_gpus = 1#cuda.device_count()
    config = load_config(config_file)
    print(f"{n_gpus} GPUs are used for inference")

    tile_dataset = get_tile_dataset(config)

    # Pad tile_dataset such that the final batch size can be divided by n_gpus.
    n_pad = MultiGPUTools.pad_size(len(tile_dataset), n_gpus, config['BASEMODEL']['Batch_Size'])
    tile_dataset = MultiGPUTools.pad_dataframe(tile_dataset, n_pad)

    L.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)
    torch.set_float32_matmul_precision('medium')

    val_transform = get_transforms(config)

    data = DataLoader(DataGenerator(tile_dataset, config=config, transform=val_transform),
              batch_size=config['BASEMODEL']['Batch_Size'],
              num_workers=int(.8 * mp.Pool()._processes),
              shuffle=False,
              pin_memory=False)    

    ddp = DDPStrategy(timeout=datetime.timedelta(seconds=14400))
    trainer = L.Trainer(devices=n_gpus,
                        accelerator="gpu",
                        #strategy='ddp',
                        benchmark=False,
                        precision=config['BASEMODEL']['Precision'],
                        callbacks=[TQDMProgressBar(refresh_rate=1)])

    model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
    model.eval()

    # Predict
    predictions = trainer.predict(model, data)
    ordered_preds = MultiGPUTools.reorder_predictions(predictions)  # reorder if processing was done on multiple GPUs
    predicted_classes_prob = torch.Tensor.cpu(torch.cat(ordered_preds))

    # Drop padding
    tile_dataset = tile_dataset.iloc[:-n_pad]
    predicted_classes_prob = predicted_classes_prob[:-n_pad]

    # Save locally (no upload to OMERO for the sarcoma classification yet)
    print('Saving sarcoma classification results locally to npy files...')

    mesenchymal_tumour_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))

    # Append tumour type probabilities to tumour tiles.
    for tumour_no, tumour_name in enumerate(mesenchymal_tumour_names):
        curkey = 'prob_' + config['DATA']['Label'] + '_' + tumour_name
        tile_dataset[curkey] = pd.Series(predicted_classes_prob[:, tumour_no], index=tile_dataset.index)
        tile_dataset = tile_dataset.fillna(0)

    for id_external, df_split in tile_dataset.groupby(tile_dataset.id_external):
        npy_path = SaveFileParameter(config, df_split, str(id_external))
        print('File exported at {}.'.format(npy_path))

    print('Done.')


if __name__ == "__main__":
    Inference(sys.argv[1])

