from PreProcessing.PreProcessingTools import PreProcessor
import toml
import pytorch_lightning as pl
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet
from Dataloader.Dataloader import *

config = toml.load(sys.argv[1])
########################################################################################################################
# 1. Download all relevant files based on the configuration file

dataset = QueryFromServer(config)
Synchronize(config, dataset)
print(dataset)

########################################################################################################################
# 2. Pre-processing: create npy files
# option #1: preprocessor + save to npy
preprocessor = PreProcessor(config)
coords_file  = preprocessor.getAllTiles(dataset)


# option #2: load/save existing
#SaveFileParameter(config, coords_file)
# coords_file = LoadFileParameter(config, dataset)

########################################################################################################################
# 3. Model
pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, precision=config['BASEMODEL']['Precision'])
model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
model.eval()

########################################################################################################################
#4. Dataloader
coords_file['SVS_PATH'] = coords_file.apply(lambda row:dataset.loc[dataset['id_internal']==row['SVS_ID']]['SVS_PATH'],axis=1) #Stitch SVS Path local to coords_file 
data = DataLoader(DataGenerator(coords_file, transform=val_transform, inference=True),
                  batch_size=config['BASEMODEL']['Batch_Size'],
                  num_workers=10,
                  shuffle=False,
                  pin_memory=True)


predictions = trainer.predict(model, data)
########################################################################################################################
#5. Save
predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))
tissue_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))

for tissue_no, tissue_name in enumerate(tissue_names):
    coords_file['prob_' + config['DATA']['Label'] + '_' + tissue_name] = pd.Series(predicted_classes_prob[:, tissue_no],index=coords_file.index)
    coords_file = coords_file.fillna(0)

########################################################################################################################
## Send back to OMERO
conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])
for SVS_ID, df_split in coords_file.groupby(df.SVS_ID):
    npy_file = SaveFileParameter(config, df, SVS_ID)

    print("\nCreating an OriginalFile and FileAnnotation")
    file_ann = conn.createFileAnnfromLocalFile(npy_file, mimetype="text/plain", desc=None)
    print("Attaching FileAnnotation to Dataset: ", "File ID:", file_ann.getId(), ",", file_ann.getFile().getName(), "Size:", file_ann.getFile().getSize())
    image.linkAnnotation(file_ann)     # link it to dataset.                                                                                                                                                           
conn.close()

