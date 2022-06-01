import os
from Dataloader.Dataloader import *
from preprocessing.PreProcessingTools import PreProcessor
import toml
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet

# Load configuration file and name
config = toml.load(sys.argv[1])

datasets = QueryFromServer(config)
Synchronize(config, datasets)
name = GetInfo.format_model_name(config)

# Set up all logging (if training)
if config['MODEL']['Inference'] is False:

    if 'logger_folder' in config['CHECKPOINT']:
        logger = TensorBoardLogger(os.path.join('lightning_logs', config['CHECKPOINT']['logger_folder']), name=name)
    else:
        logger = TensorBoardLogger('lightning_logs', name=name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath     =config['MODEL']['Model_Save_Path'],
        monitor     =config['CHECKPOINT']['Monitor'],
        filename    =name + '-epoch{epoch:02d}-' + config['CHECKPOINT']['Monitor'] + '{' + config['CHECKPOINT']['Monitor'] + ':.2f}',
        save_top_k  =1,
        mode        =config['CHECKPOINT']['Mode'])

pl.seed_everything(config['MODEL']['Random_Seed'], workers=True)

# Load coords_file
coords_file = LoadFileParameter(config, datasets)

# Augment data on the training set
if config['AUGMENTATION']['Rand_Operations'] > 0:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=config['AUGMENTATION']['Rand_Operations'], magnitude=config['AUGMENTATION']['Rand_Magnitude']),  # this only operates on 8-bit images (not normalised float32 tensors)
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# No data augmentation on the validation settens
val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


if config['MODEL']['Inference'] is False:  # train
    data = DataModule(
        coords_file,
        batch_size=config['MODEL']['Batch_Size'],
        train_transform=train_transform,
        val_transform=val_transform,
        train_size=config['DATA']['Train_Size'],
        val_size=config['DATA']['Val_Size'],
        inference=False,
        dim_list=config['DATA']['Patch_Size'],
        vis_list=config['DATA']['Vis'],
        n_per_sample=config['DATA']['N_Per_Sample'],
        target=config['DATA']['Label'],
        sampling_scheme=config['DATA']['Sampling_Scheme']
    )
    config['DATA']['N_Training_Examples'] = data.train_data.__len__()

else:  # prediction does not use train/validation sets, only directly the dataloader.
    data = DataLoader(DataGenerator(coords_file, transform=val_transform, inference=True),
                      batch_size=config['MODEL']['Batch_Size'],
                      num_workers=10,
                      shuffle=False,
                      pin_memory=True)

# Return some stats/information on the training/validation data (to explore the dataset / sanity check)
# From paper: Class-balanced Loss Based on Effective Number of Samples
config['INTERNAL']['weights'] = torch.ones(int(config['DATA']['N_Classes'])).float()
if config['MODEL']['Inference'] is False:
    npatches_per_class = GetInfo.ShowTrainValTestInfo(data, config)

    # The following will be used in an upcoming release to add weights to labels. This will be packaged in a function:
    # N = sum(npatches_per_class)
    # beta = (N-1)/N
    # effective_samples = (1 - beta**npatches_per_class)/(1-beta)
    # raw_scores = 1 / effective_samples
    # w = config['DATA']['N_Classes'] * raw_scores / sum(raw_scores)
    # config['INTERNAL']['weights'] = torch.tensor(w).float()
    # print(config['INTERNAL']['weights'])

# Load model and train/infer
if config['MODEL']['Inference'] is False:  # train

    trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=config['MODEL']['Max_Epochs'],
                         precision=config['MODEL']['Precision'], callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger)

    if config['MODEL']['Base_Model'].lower() == 'convnet':
        model = ConvNet(config)
    elif config['MODEL']['Base_Model'].lower() == 'convnext':
        model = ConvNeXt(config)
    elif config['MODEL']['Base_Model'].lower() == 'vit':
        model = ViT(config)
    else:
        raise RuntimeError('No existing model associated with "' + config['MODEL']['Base_Model'] + '".')

    trainer.fit(model, data)

else:  # infer

    trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, precision=config['MODEL']['Precision'])

    if config['MODEL']['Base_Model'].lower() == 'convnet':
        model = ConvNet.load_from_checkpoint(config=config, checkpoint_path=config['MODEL']['Model_Save_Path'])
    elif config['MODEL']['Base_Model'].lower() == 'convnext':
        model = ConvNeXt.load_from_checkpoint(config=config, checkpoint_path=config['MODEL']['Model_Save_Path'])
    elif config['MODEL']['Base_Model'].lower() == 'vit':
        model = ViT.load_from_checkpoint(config=config, checkpoint_path=config['MODEL']['Model_Save_Path'])
    else:
        raise RuntimeError('No existing model associated with "' + config['MODEL']['Base_Model'] + '".')

    model.eval()
    predictions = trainer.predict(model, data)
    predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))

    for i in range(predicted_classes_prob.shape[1]):
        print('Adding the column ' + '"prob_' + config['DATA']['Label'] + str(i) + '"...')
        SaveFileParameter(config, coords_file, predicted_classes_prob[:, i],
                          'prob_' + config['DATA']['Label'] + str(i))



