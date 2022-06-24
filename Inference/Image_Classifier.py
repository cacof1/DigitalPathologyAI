from PreProcessing.PreProcessingTools import PreProcessor
import toml
import pytorch_lightning as pl
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet
from Dataloader.Dataloader import *

# config = toml.load(sys.argv[1])
config = toml.load('../Configs/sarcoma/infer_sarcoma_convnet_10classes.ini')

########################################################################################################################
# 1. Download all relevant files based on the configuration file

dataset = QueryFromServer(config)
Synchronize(config, dataset)
print(dataset)

########################################################################################################################
# 2. Pre-processing: create npy files

# Load pre-processed dataset. It should have been pre-processed with Inference/Preprocess.py first.
coords_file = LoadFileParameter(config, dataset)

# Mask the coords_file to only keep the tumour tiles, depending on a pre-set criteria.
coords_file = coords_file[coords_file['prob_tissue_type_tumour'] > 0.85]

# Replace coords file path
list_ids = coords_file['SVS_PATH'].unique()
for index, row in dataset.iterrows():
    match = [row['id_external'] in cid for cid in list_ids]
    coords_file_SVS_path = list_ids[np.argwhere(match)[0][0]]
    mask = coords_file.SVS_PATH == coords_file_SVS_path
    coords_file.loc[mask, 'SVS_PATH'] = row.SVS_PATH

########################################################################################################################
# 3. Model evaluation

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    transforms.Lambda(lambda x: x * 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    transforms.Lambda(lambda x: x / 255) if 'Colour_Norm_File' in config['NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = DataLoader(DataGenerator(coords_file, transform=val_transform, inference=True),
                  batch_size=config['BASEMODEL']['Batch_Size'],
                  num_workers=10,
                  shuffle=False,
                  pin_memory=True)

trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, precision=config['BASEMODEL']['Precision'])
model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])

model.eval()
predictions = trainer.predict(model, data)
predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))

mesenchymal_tumour_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))

for tumour_no, tumour_name in enumerate(mesenchymal_tumour_names):
    coords_file['prob_' + config['DATA']['Label'] + '_' + tumour_name] = pd.Series(predicted_classes_prob[:, tumour_no],
                                                                                   index=coords_file.index)
    coords_file = coords_file.fillna(0)

SaveFileParameter(config, coords_file)
