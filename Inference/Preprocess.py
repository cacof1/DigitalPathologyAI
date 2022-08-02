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
print(SVS_dataset)

########################################################################################################################
# 2. Pre-processing: create npy files

preprocessor = PreProcessor(config)
tile_dataset = preprocessor.getAllTiles(SVS_dataset)

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
# 5. Save

tissue_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))
for tissue_no, tissue_name in enumerate(tissue_names):
    
    tile_dataset['prob_' + config['DATA']['Label'] + '_' + tissue_name] = predicted_classes_prob[:, tissue_no]
    tile_dataset = tile_dataset.fillna(0)

# todo: remove both following lines once the SaveFileParameter below works.
for SVS_ID, df_split in tile_dataset.groupby(tile_dataset.SVS_ID):
    npy_file = SaveFileParameter(config, df_split, SVS_ID)

########################################################################################################################
# 6. Send back to OMERO
conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])
conn.SERVICE_OPTS.setOmeroGroup('-1')

for SVS_ID, df_split in tile_dataset.groupby(tile_dataset.SVS_ID):
    image = conn.getObject("Image", SVS_dataset.loc[SVS_dataset["id_internal"] == SVS_ID].iloc[0]['id_omero'])
    group_id = image.getDetails().getGroup().getId()
    conn.SERVICE_OPTS.setOmeroGroup(group_id)
    print("Current group: ", group_id)
    npy_file = SaveFileParameter(config, df_split, SVS_ID)
    print("\nCreating an OriginalFile and FileAnnotation")
    file_ann = conn.createFileAnnfromLocalFile(npy_file, mimetype="text/plain", desc=None)
    print("Attaching FileAnnotation to Dataset: ", "File ID:", file_ann.getId(), ",", file_ann.getFile().getName(),
          "Size:", file_ann.getFile().getSize())

    ## delete because Omero methods are moronic
    to_delete = []
    for ann in image.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper): to_delete.append(ann.id)
        conn.deleteObjects('Annotation', to_delete, wait=True)
    if len(to_delete)>0: image.linkAnnotation(file_ann)  # link it to dataset.
    
    print('{}.npy uploaded'.format(SVS_ID))
    
conn.close()
