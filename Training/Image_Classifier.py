import sys
import os
import datetime
import torch
from torch import cuda
from torchvision import transforms
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
import toml
from sklearn import preprocessing
from lightning.pytorch.strategies import DDPStrategy
from Dataloader.Dataloader import (
    DataModule,
    LoadFileParameter,
)
from Utils import GetInfo
from Utils.OmeroTools import (
    QueryImageFromCriteria,
    SynchronizeSVS,
    SynchronizeNPY
)
from Model.ConvNet import ConvNet
from QA.Normalization.Colour import ColourAugment
import datetime, time
from pytorch_lightning.profiler import PyTorchProfiler
def load_config(config_file):
    return toml.load(config_file)

def get_tile_dataset(config):
    SVS_dataset = QueryImageFromCriteria(config)
    SVS_dataset = SVS_dataset.groupby('diagnosis').head(n=5)
    SynchronizeSVS(config, SVS_dataset)    
    SynchronizeNPY(config, SVS_dataset)
    tile_dataset = LoadFileParameter(config, SVS_dataset)
    tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_Tumour'] > 0.94]
    tile_dataset = tile_dataset.merge(SVS_dataset, on='id_external')
    tile_dataset['SVS_PATH'] = tile_dataset['SVS_PATH_y']
    return tile_dataset

def get_logger(config, model_name):
    logger_folder = config['CHECKPOINT']['logger_folder']
    timestamp     = str(round(time.time()))
    return TensorBoardLogger('lightning_logs', name=logger_folder, sub_dir=model_name)#, default_hp_metric=False)

def get_callbacks(config):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor   = config['CHECKPOINT']['Monitor'],
        filename  = f"{{model_name}}-epoch{{epoch:02d}}",
        save_top_k= 1,
        mode      = config['CHECKPOINT']['Mode'])

    return [lr_monitor, checkpoint_callback]

def get_transforms(config):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        ColourAugment.ColourAugment(sigma=config['AUGMENTATION']['Colour_Sigma'], mode=config['AUGMENTATION']['Colour_Mode']),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def main(config_file):
    n_gpus = 1#cuda.device_count()
    config = load_config(config_file)
    print(f"{n_gpus} GPUs are used for training")

    tile_dataset = get_tile_dataset(config)
    config['DATA']['N_Classes'] = len(tile_dataset[config['DATA']['Label']].unique())

    model_name = GetInfo.format_model_name(config)
    logger     = get_logger(config, model_name)
    callbacks  = get_callbacks(config)

    L.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)
    torch.set_float32_matmul_precision('medium')
    
    train_transform, val_transform = get_transforms(config)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(tile_dataset[config['DATA']['Label']])
    #ddp = DDPStrategy(timeout=datetime.timedelta(seconds=7200))
    trainer = L.Trainer(devices=1,
                        accelerator="gpu",
                        max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                        precision=config['BASEMODEL']['Precision'],
                        callbacks=callbacks,
                        log_every_n_steps=50,
                        logger=logger)

    model = ConvNet(config, label_encoder=label_encoder)
    
    data = DataModule(
        tile_dataset,
        train_transform=train_transform,
        val_transform=val_transform,
        config= config,
        label_encoder=label_encoder
    )
    
    #GetInfo.ShowTrainValTestInfo(data, config)

    trainer.fit(model, data)
    trainer.test(ckpt_path='best', dataloaders=data.test_dataloader())

    ## Save the Split
    data.tile_dataset_train.to_csv(logger.log_dir + "/train_set.csv")
    data.tile_dataset_val.to_csv(logger.log_dir + "/val_set.csv")
    data.tile_dataset_test.to_csv(logger.log_dir + "/test_set.csv")        

    with open(logger.log_dir + "/Config.ini", "w+") as toml_file:
        toml.dump(config, toml_file)
        toml_file.write("Train transform:\n")
        toml_file.write(str(train_transform))
        toml_file.write("Val/Test transform:\n")
        toml_file.write(str(val_transform))

if __name__ == "__main__":
    main(sys.argv[1])
