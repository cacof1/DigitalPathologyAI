import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
from PreProcessing.PreProcessingTools import PreProcessor
import toml
import pytorch_lightning as pl
from torchvision import transforms
from QA.Normalization.Colour import ColourNorm
from Model.ConvNet import ConvNet
from Dataloader.ObjectDetection import *
from Utils import MultiGPUTools
import pandas as pd
import numpy as np

n_gpus = torch.cuda.device_count()  # could go into config file
config = toml.load('/home/dgs2/Software/DigitalPathologyAI/Configs/config_mitotic_classification_inference.ini')#sys.argv[1]
########################################################################################################################
diagnosis = config['DATA']['Diagnosis']
Inference_df = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/Inference_{}.csv'.format(diagnosis))
SVS_IDs = Inference_df.id_internal.unique()

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)

val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
model.eval()
print('Model loaded: {}'.format(config['CHECKPOINT']['Model_Save_Path']))

df_list = []

for count, SVS_ID in enumerate(SVS_IDs):
    try:
        slidedataset = pd.read_csv(config['DATA']['Detection_Path']+'{}_detected_coords.csv'.format(SVS_ID))
    except:
        print('{} Not Found'.format(SVS_ID))
        continue

    p = pathlib.Path(config['DATA']['Detection_Path'] + '{}_classification_coords.csv'.format(SVS_ID))
    if os.path.isfile(p):
        delta = datetime.now() - datetime.fromtimestamp(p.stat().st_ctime)
        time_difference = delta.days
    else:
        time_difference = 9999

    if time_difference < 10:
        print('{}: {} is already compeleted on {}'.format(count, SVS_ID, datetime.fromtimestamp(p.stat().st_ctime)))
        continue

    slidedataset['SVS_ID'] = [SVS_ID] * slidedataset.shape[0]
    slidedataset['index'] = slidedataset.index
    #slidedataset = slidedataset[slidedataset['scores'] > 0.7]
    slidedataset = slidedataset[slidedataset['prob_tissue_type_Tumour'] > 0.94].reset_index(drop=True)
    print('Processing {}/{}: {}'.format(count, len(SVS_IDs), SVS_ID))
    print(slidedataset)

    n_pad = MultiGPUTools.pad_size(len(slidedataset), n_gpus, config['BASEMODEL']['Batch_Size'])
    slidedataset = MultiGPUTools.pad_dataframe(slidedataset, n_pad)

    data = DataLoader(MixDataset(slidedataset,
                                 masked_input=config['DATA']['masked_input'],
                                 wsi_folder=config['DATA']['SVS_Folder'],
                                 mask_folder=config['DATA']['Mask_Folder'],
                                 data_source=config['DATA']['data_source'],
                                 dim=(64, 64),
                                 vis_level=0,
                                 channels=3,
                                 transform=val_transform,
                                 inference=True),
                      num_workers=config['BASEMODEL']['Num_of_Worker'],
                      persistent_workers=True,
                      batch_size=config['BASEMODEL']['Batch_Size'],
                      shuffle=False,
                      pin_memory=True,)

    trainer = pl.Trainer(accelerator='gpu', devices=[0,1,2,3],
                         strategy='ddp_find_unused_parameters_false',
                         benchmark=True,
                         precision=config['BASEMODEL']['Precision'],
                         callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=1)])

    predictions = trainer.predict(model, data)
    predictions = MultiGPUTools.reorder_predictions(predictions)  # reorder if processing was done on multiple GPUs
    predicted_classes_prob = torch.Tensor.cpu(torch.cat(predictions))

    slidedataset = slidedataset.iloc[:-n_pad]
    predicted_classes_prob = predicted_classes_prob[:-n_pad]

    classes = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))
    for i, class_name in enumerate(classes):
        slidedataset['prob_' + str(class_name)] = pd.Series(predicted_classes_prob[:, i], index=slidedataset.index)
        tile_dataset = slidedataset.fillna(0)

    slidedataset.to_csv(config['DATA']['Detection_Path'] + '{}_classification_coords.csv'.format(SVS_ID),index=False)
    df_list.append(slidedataset)

df_all = pd.concat(df_list)
df_all.to_csv(config['DATA']['Detection_Path'] + 'classification_coords_{}.csv'.format(diagnosis), index=False)











