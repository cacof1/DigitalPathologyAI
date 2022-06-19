from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import openslide
import torch
from collections import Counter
#import npyExportTools
import itertools
import Utils.sampling_schemes as sampling_schemes
from Utils.OmeroTools import *
from Utils import npyExportTools
from pathlib import Path

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, coords_file, target="tumour_label", dim_list=[(256, 256)], vis_list=[0], inference=False, transform=None, target_transform=None):

        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.coords = coords_file
        self.vis_list = vis_list
        self.dim_list = dim_list
        self.inference = inference
        self.target = target

    def __len__(self):
        return int(self.coords.shape[0])

    def __getitem__(self, id):
        # load image
        wsi_file = openslide.open_slide(self.coords["SVS_PATH"].iloc[id])

        data_dict = {}
        for dim in self.dim_list:
            for vis_level in self.vis_list:
                key = "_".join(map(str, dim)) + "_" + str(vis_level)
                data_dict[key] = np.array(
                    wsi_file.read_region([self.coords["coords_x"].iloc[id], self.coords["coords_y"].iloc[id]],
                                         vis_level, dim).convert("RGB"))

        ## Transform - Data Augmentation

        if self.transform: data_dict = {key: self.transform(value) for (key, value) in data_dict.items()}
        if (self.inference):
            return data_dict

        else:
            label = self.coords[self.target].iloc[id]
            if self.target_transform:
                label = self.target_transform(label)
            return data_dict, label


class DataModule(LightningDataModule):

    def __init__(self, coords_file, train_transform=None, val_transform=None, batch_size=8, n_per_sample=np.Inf,
                 train_size=0.7, val_size=0.3, target=None, sampling_scheme='wsi', label_encoder=None, **kwargs):
        super().__init__()

        
        self.batch_size = batch_size
        coords_file[target] = label_encoder.transform(coords_file[target])

        if sampling_scheme.lower() == 'wsi':
            coords_file_sampled = sampling_schemes.sample_N_per_WSI(coords_file, n_per_sample=n_per_sample)
            svi = np.unique(coords_file_sampled.SVS_PATH)
            np.random.shuffle(svi)
            train_idx, val_idx = train_test_split(svi, test_size=val_size, train_size=train_size)
            coords_file_train = coords_file_sampled[coords_file_sampled.SVS_PATH.isin(train_idx)]
            coords_file_valid = coords_file_sampled[coords_file_sampled.SVS_PATH.isin(val_idx)]

        elif sampling_scheme.lower() == 'patch':
            coords_file_sampled = sampling_schemes.sample_N_per_WSI(coords_file, n_per_sample=n_per_sample)
            coords_file_train, coords_file_valid = train_test_split(coords_file_sampled,
                                                                    test_size=val_size, train_size=train_size)

        else:  # assume custom split
            sampler = getattr(sampling_schemes, sampling_scheme)
            coords_file_train, coords_file_valid = sampler(coords_file, target=target, n_per_sample=n_per_sample,
                                                           train_size=train_size, test_size=val_size)

        self.train_data = DataGenerator(coords_file_train, transform=train_transform, target=target, **kwargs)
        self.val_data   = DataGenerator(coords_file_valid, transform=val_transform, target=target, **kwargs)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=10, pin_memory=True)

def LoadFileParameter(config, dataset):

    cur_basemodel_str = npyExportTools.basemodel_to_str(config)
    coords_file = pd.DataFrame()

    for svs_path in dataset.SVS_PATH:
        npy_path    = os.path.join(os.path.split(svs_path)[0], 'patches', os.path.split(svs_path)[1].replace('svs', 'npy'))
        existing_df = np.load(npy_path, allow_pickle=True).item()[cur_basemodel_str][1]
        coords_file = pd.concat([coords_file, existing_df], ignore_index=True)

    return coords_file

def SaveFileParameter(config, df, SVS_ID):

    cur_basemodel_str = npyExportTools.basemodel_to_str(config)
    npy_path = os.path.join(config['DATA']['SVS_Folder'], 'patches', SVS_ID+".npy")
    os.makedirs(os.path.split(npy_path)[0], exist_ok=True)  # in case folder is non-existent
    npy_dict = np.load(npy_path, allow_pickle=True).item() if os.path.exists(npy_path) else {}
    npy_dict[cur_basemodel_str] = [config, df]
    np.save(npy_path, npy_dict)
    return npy_path

def QueryFromServer(config, **kwargs):
    print("Querying from Server")
    df   = pd.DataFrame()
    conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])  ## Group not implemented yet
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
            if nb == 0: query_end += "where (mv" + str(nb) + ".name = :key" + str(nb) + " and mv" + str(nb) + ".value = :value" + str(nb) + ")"
            else:       query_end += " and (mv" + str(nb) + ".name = :key" + str(nb) + " and mv" + str(nb) + ".value = :value" + str(nb) + ")"
            params.addString('key' + str(nb), keys[nb])
            params.addString('value' + str(nb), temp)

        query   = query_base + query_end
        result  = conn.getQueryService().projection(query, params, {"omero.group": "-1"})

        ## Version 1  -- populate only the criteria
        """
        df_criteria = pd.DataFrame([[row[0].val, Path(row[1].val).stem, row[2].val] for row in result], columns=["id_omero", "id_external", "Size"])                                  
        for nb, temp in enumerate(value): df_criteria[keys[nb]]   = temp
        """

        ## Version 2 -- populate everything (fragile)
        df_criteria = pd.DataFrame()            
        for row in result: ## Transform the results into a panda dataframe for each found match
            temp = pd.DataFrame([[row[0].val, Path(row[1].val).stem, row[2].val, *row[3].val.getMapValueAsMap().values()] for row in result],
                                columns=["id_omero", "id_external", "Size", *row[3].val.getMapValueAsMap().keys()])
            df_criteria = pd.concat([df_criteria, temp])                                    


        df_criteria['SVS_PATH'] = [os.path.join(config['DATA']['SVS_Folder'], image_id+'.svs') for image_id in df_criteria['id_external']]
        df = pd.concat([df, df_criteria])
        
    conn.close()
    return df

def Synchronize(config, df):
    conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])
    for index, image in df.iterrows():
        filepath = Path(config['DATA']['SVS_Folder'], image['id_external']+'.svs')  
        if filepath.is_file():  # Exist
            if not filepath.stat().st_size == image['Size']:  # Corrupted
                print("File size doesn't match, redownloading")
                os.remove(filepath)
                download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])
                download_annotation(conn.getObject("Image", image['id_omero']), config['DATA']['SVS_Folder'])
        else:  ## Doesn't exist
            print("Doesn't exist")
            download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])
            download_annotation(conn.getObject("Image", image['id_omero']), config['DATA']['SVS_Folder'])
    conn.close()
            
