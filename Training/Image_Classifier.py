from torchvision import transforms
import torch
import pytorch_lightning as pl
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataModule, WSIQuery, DataGenerator
from Model.ImageClassifier import ImageClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import toml
from utils import GetInfo
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, confusion_matrix
import numpy as np

# Load configuration file and name
config = toml.load('SarcomaTrainer.ini')
name = GetInfo.format_model_name(config)

# Set up all logging
logger = TensorBoardLogger('lightning_logs', name=name)
checkpoint_callback = ModelCheckpoint(
    dirpath     =config['MODEL']['Model_Save_Path'],
    monitor     =config['CHECKPOINT']['monitor'],
    filename    =name + '-epoch{epoch:02d}-' + config['CHECKPOINT']['monitor'] + '{' + config['CHECKPOINT']['monitor'] + ':.2f}',
    save_top_k  =1,
    mode        =config['CHECKPOINT']['mode'])

pl.seed_everything(config['MODEL']['RANDOM_SEED'], workers=True)

# Return WSI according to the selected CRITERIA in the configuration file.
ids = WSIQuery(config)

if config['DATA']['target'] == 'sarcoma_label':  # TODO : potentially move the following step out of Image_Classifier
    # Specific to sarcoma study: make sure that all ids have their "sarcoma_label" target.
    # For another target, make sure you use your own function to append your targets to csv files.
    from __local.SarcomaClassification.Methods import AppendSarcomaLabel
    AppendSarcomaLabel(ids, config['DATA']['SVS_Folder'], config['DATA']['Patches_Folder'],
                       mapping_file='mapping_SFTl_DF_NF_SF')

# Load coords_file
coords_file = LoadFileParameter(ids, config['DATA']['SVS_Folder'], config['DATA']['Patches_Folder'])

if config['DATA']['target'] == 'sarcoma_label':  # TODO: maybe encode more efficiently in the config file.
    # Select a subset of coords files. In the sarcoma study, we only consider patches labelled as tumour.
    coords_file = coords_file[coords_file["tumour_pred_label_1"] > coords_file["tumour_pred_label_0"]]

transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if config['MODEL']['inference'] is False:  # train
    data = DataModule(
        coords_file,
        batch_size=config['MODEL']['Batch_Size'],
        train_transform=transform,
        val_transform=transform,
        inference=False,
        dim_list=config['DATA']['dim'],
        vis_list=config['DATA']['vis'],
        n_per_sample=config['DATA']['n_per_sample'],
        target=config['DATA']['target']
    )
else:  # prediction does not use train/validation sets, only directly the dataloader.
    data = DataLoader(DataGenerator(coords_file, transform=transform, inference=True),
                      batch_size=config['MODEL']['Batch_Size'],
                      num_workers=10,
                      shuffle=False,
                      pin_memory=True)

# Return some stats/information on the training/validation data (to explore the dataset / sanity check)
GetInfo.ShowTrainValTestInfo(data, config)

# Load model and train/infer
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=config['MODEL']['Max_Epochs'],
                     precision=config['MODEL']['Precision'], callbacks=[checkpoint_callback], logger=logger)

if config['MODEL']['inference'] is False:  # train
    model = ImageClassifier(config)
    trainer.fit(model, data)
else:  # infer
    model = ImageClassifier.load_from_checkpoint(config=config, checkpoint_path=config['MODEL']['Model_Save_Path'])
    model.eval()
    predictions = trainer.predict(model, data)
    predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))
    for i in range(predicted_classes_prob.shape[1]):
        SaveFileParameter(coords_file, config['DATA']['Patches_Folder'], predicted_classes_prob[:, i],
                          'sarcoma_pred_label_' + str(i))  ## TODO: fix to be more general.
    preds = torch.argmax(predicted_classes_prob, dim=1)
    targets = torch.tensor(data.dataset.coords.sarcoma_label.values.astype(int))
    final_acc = accuracy(preds, targets)
    print('Final accuracy over entire dataset is: {}'.format(final_acc))
    CF = confusion_matrix(preds, targets, config['DATA']['n_classes'])
    print('Confusion matrix:')
    print(CF)
    print('------------------')

    # Statistics per SVS.
    file_ids = data.dataset.coords.file_id.unique()
    acc_per_SVS = list()
    for file_id in file_ids:
        mask = data.dataset.coords.file_id == file_id
        #print(sum(mask))
        cur_targets = torch.tensor(data.dataset.coords.sarcoma_label.values[mask].astype(int))
        cur_preds = preds[mask.values]
        the_acc = accuracy(cur_preds, cur_targets)
        #the_acc = torch.mode(cur_preds).values == torch.mode(cur_targets).values  # if voting per mode instead
        acc_per_SVS.append(the_acc)
        the_cur_label = np.unique(data.dataset.coords.sarcoma_label[mask])
        print('{}, {}, {}'.format(file_id, the_cur_label[0], 100*the_acc.numpy()))
