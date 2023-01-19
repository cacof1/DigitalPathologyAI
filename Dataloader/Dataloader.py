from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn import preprocessing
import openslide
import torch
from collections import Counter
import itertools
import Utils.sampling_schemes as sampling_schemes
from Utils.OmeroTools import *
from Utils import npyExportTools
from pathlib import Path
from QA.Normalization.Colour import ColourNorm

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, tile_dataset, target="tumour_label", dim = (256, 256), vis = 0, inference=False,
                 transform=None, target_transform=None, svs_folder=None):

        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.tile_dataset = tile_dataset
        self.vis       = vis
        self.dim       = dim
        self.inference = inference
        self.target = target
        self.svs_folder = svs_folder

    def __len__(self):
        return int(self.tile_dataset.shape[0])

    def __getitem__(self, id):
        # load image
        svs_path = self.tile_dataset['SVS_PATH'].iloc[id]
        svs_file = openslide.open_slide(svs_path)
        data = svs_file.read_region([self.tile_dataset["coords_x"].iloc[id], self.tile_dataset["coords_y"].iloc[id]], self.vis, self.dim).convert("RGB")
        
        for transform_step in self.transform.transforms:
            if(isinstance(transform_step,ColourNorm.Macenko)):
                HE, maxC = ColourNorm.Macenko().find_HE(data, get_maxC=True)
                #data     = transform_step(data, self.tile_dataset['HE'].iloc[id])
                data     = transform_step(data, HE, maxC)
                continue
            else:
                data = transform_step(data)

        ## Transform - Data Augmentation
        #if self.transform: data = self.transform(data)

        if self.inference:
            return data

        else: ## Training
            label = self.tile_dataset[self.target].iloc[id]
            if self.target_transform: label = self.target_transform(label)

            return data, label


class DataModule(LightningDataModule):

    def __init__(self, tile_dataset, train_transform=None, val_transform=None, batch_size=8, n_per_sample=np.Inf,
                 train_size=0.7, test_size=0.15, target=None,
                 label_encoder=None, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        if(label_encoder):  tile_dataset[target] = label_encoder.transform(tile_dataset[target])  ## For classif only


        # Split Test with differents WSI
        # Code grabs test_size % of all data for testing. Then the remaining is split as
        gss                       = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_val_inds, test_inds = next(gss.split(tile_dataset, groups=tile_dataset['id_external']))
        tile_dataset_train_val    = tile_dataset.iloc[train_val_inds].reset_index()
        tile_dataset_test         = tile_dataset.iloc[test_inds].reset_index()

        # Select only N per sample from each class
        tile_dataset_train_val   = tile_dataset_train_val.groupby(target).head(n=n_per_sample)
        tile_dataset_test        = tile_dataset_test.groupby(target).head(n=n_per_sample)

        # Shuffle Train and Val
        tile_dataset_train, tile_dataset_val = train_test_split(tile_dataset_train_val,
                                                                train_size=train_size,
                                                                stratify=tile_dataset_train_val[target],
                                                                random_state=np.random.randint(10000))

        self.train_data = DataGenerator(tile_dataset_train, transform=train_transform, target=target, **kwargs)
        self.val_data   = DataGenerator(tile_dataset_val, transform=val_transform, target=target, **kwargs)
        self.test_data = DataGenerator(tile_dataset_test, transform=val_transform, target=target, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=60, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=60, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=60, pin_memory=True)


def LoadFileParameter(config, dataset):

    cur_basemodel_str = npyExportTools.basemodel_to_str(config)
    tile_dataset = pd.DataFrame()
    for npy_path in dataset.NPY_PATH:
        header, existing_df = np.load(npy_path, allow_pickle=True).item()[cur_basemodel_str]
        tile_dataset = pd.concat([tile_dataset, existing_df], ignore_index=True)
    return tile_dataset


def SaveFileParameter(config, df, SVS_ID):
    cur_basemodel_str = npyExportTools.basemodel_to_str(config)
    npy_path = os.path.join(config['DATA']['SVS_Folder'], 'patches', SVS_ID + ".npy")
    os.makedirs(os.path.split(npy_path)[0], exist_ok=True)  # in case folder is non-existent
    npy_dict = np.load(npy_path, allow_pickle=True).item() if os.path.exists(npy_path) else {}
    npy_dict[cur_basemodel_str] = [config, df]
    np.save(npy_path, npy_dict)
    return npy_path


def QueryImageFromCriteria(config, **kwargs):
    print("Querying from Server")
    df = pd.DataFrame()
    conn = connect(config['OMERO']['Host'], config['OMERO']['User'],
                   config['OMERO']['Pw'])  ## Group not implemented yet
    conn.SERVICE_OPTS.setOmeroGroup('-1')

    keys = list(config['CRITERIA'].keys())
    value_iter = itertools.product(*config['CRITERIA'].values())  ## Create a joint list with all elements
    for value in value_iter:
        query_base = """
        select image.id, image.name, f2.size, a from
        ImageAnnotationLink ial
        join ial.child a
        join ial.parent image
        left outer join image.pixels p
        left outer join image.fileset as fs
        left outer join fs.usedFiles as uf
        left outer join uf.originalFile as f2
        """
        query_end = ""
        params = omero.sys.ParametersI()
        for nb, temp in enumerate(value):
            query_base += "join a.mapValue mv" + str(nb) + " \n        "
            if nb == 0:
                query_end += "where (mv" + str(nb) + ".name = :key" + str(nb) + " and mv" + str(
                    nb) + ".value = :value" + str(nb) + ")"
            else:
                query_end += " and (mv" + str(nb) + ".name = :key" + str(nb) + " and mv" + str(
                    nb) + ".value = :value" + str(nb) + ")"
            params.addString('key' + str(nb), keys[nb])
            params.addString('value' + str(nb), temp)

        query = query_base + query_end
        result = conn.getQueryService().projection(query, params, {"omero.group": "-1"})

        df_criteria = pd.DataFrame()
        for row in result:  ## Transform the results into a panda dataframe for each found match
            temp = pd.DataFrame(
                [[row[0].val, Path(row[1].val).stem, row[2].val, *row[3].val.getMapValueAsMap().values()]],
                columns=["id_omero", "id_external", "Size", *row[3].val.getMapValueAsMap().keys()])
            df_criteria = pd.concat([df_criteria, temp])

        df_criteria['SVS_PATH'] = [os.path.join(config['DATA']['SVS_Folder'], image_id + '.svs') for image_id in
                                   df_criteria['id_external']]
        df_criteria['NPY_PATH'] = [os.path.join(config['DATA']['SVS_Folder'], 'patches', image_id + '.npy') for image_id
                                   in df_criteria['id_external']]

        df = pd.concat([df, df_criteria])

    conn.close()
    return df

def SynchronizeSVS(config, df):

    conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])
    conn.SERVICE_OPTS.setOmeroGroup('-1')

    for index, image in df.iterrows():
        filepath = image['SVS_PATH']
        if os.path.exists(filepath):  # Exist
            if not os.path.getsize(filepath) == image['Size']:  # Corrupted
                print("SVS file size does not match - redownloading...")

                os.remove(filepath)
                download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])
                
        else:  ## Doesn't exist
            print("SVS file does not exist - downloading...")
            download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])

    conn.close()

            
def DownloadNPY(config, df):
    conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    
    for index, image in df.iterrows(): 
        if not os.path.exists(image['NPY_PATH']):  # Doesn't exist

            npy_path = os.path.join(config['DATA']['SVS_Folder'], 'patches')
            os.makedirs(npy_path, exist_ok=True)
            download_annotation(conn.getObject("Image", image['id_omero']), npy_path)
    conn.close()
