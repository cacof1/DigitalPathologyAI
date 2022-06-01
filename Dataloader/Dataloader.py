from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import openslide
import torch
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
                 train_size=0.7, val_size=0.3, target=None, sampling_scheme='wsi', **kwargs):
        super().__init__()
        self.batch_size = batch_size

        le = preprocessing.LabelEncoder()
        coords_file[target] = le.fit_transform(coords_file[target])
        
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
        self.val_data = DataGenerator(coords_file_valid, transform=val_transform, target=target, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=10, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=10, pin_memory=True)


def gather_WSI_npy_indexes(config, npy_file):#, overwrite=True, verbose=False):
    
    if(npy_file['header']['BASEMODEL'] == config['BASEMODEL']): return npy_file['coords']
    else: sys.exit("Can't find a matching header file")
    return WSI_processing_index, processing_flag


    """
    # Locates the index from the .npy file that will be used to store the results of current session.
    WSI_processing_index = []
    processing_flag = []

    # Identify session type
    preprocessing_session = False
    if config['DATA']['Label'].lower() == 'tissue_type':
        if verbose:
            print('Assuming pre-processing session.')
        preprocessing_session = True
    else:
        if verbose:
            print('Assuming conventional session.')

    for idx in range(len(ids)):
        ID = ids[idx]
        patch_npy_export_filename = os.path.join(config['DATA']['SVS_Folder'], 'patches', str(ID) + '.npy')

        # Does the .npy file exist?
        if not os.path.exists(patch_npy_export_filename):

            WSI_processing_index.append(0)
            processing_flag.append(True)
            if verbose:
                print('WSI {}.npy: file does not exist, creating new.'.format(ID))
        else:

            # Is this a preprocessing session?
            if preprocessing_session:

                # Does the WSI npy have processing which match criteria in the config file?
                datasets = list(np.load(patch_npy_export_filename, allow_pickle=True))

                # datasets is a list, where each element contains a dictionary which holds the following two keys:
                # header: the config file (also a dictionary) used to generate a preprocessing/analysis
                # dataframe: a dataframe containing the WSI information obtained after processing using the config
                # file of the above header.

                # Next step is to find out if the current processing session already exists. It is assumed that,
                # for pre-processing, the session already exists if the config file is the same, with the exception
                # of the [CRITERIA], [VERBOSE], [CONTOURS], [OMERO] and [INTERNAL] fields. This comparison is
                # achieved below.

                header_key_blacklist = ['CRITERIA', 'VERBOSE', 'CONTOURS', 'OMERO', 'INTERNAL','PREPROCESSING_MAPPING']
                cur_reduced_header = npyExportTools.remove_dict_keys(config, header_key_blacklist)
                reduced_existing_headers = [npyExportTools.remove_dict_keys(dataset['header'], header_key_blacklist) for dataset in datasets]
                matching_header = [npyExportTools.compare_dicts(cur_reduced_header, h) for h in reduced_existing_headers]
                match = np.argwhere(matching_header)

                if len(match) == 0:  # then no, and dataset will be processed
                    WSI_processing_index.append(len(datasets))
                    processing_flag.append(True)
                    if verbose:
                        print('WSI {}.npy: preprocessing session, file exists, will append new dataset.'.format(ID))

                elif len(match) == 1:

                    WSI_processing_index.append(match[0][0])
                    # Does the user want to override the existing dataset?
                    if overwrite:
                        processing_flag.append(True)
                        if verbose:
                            print('WSI {}.npy: preprocessing session, file and dataset exists, will overwrite existing dataset.'.format(ID))
                    else:
                        processing_flag.append(False)
                        if verbose:
                            print('WSI {}.npy: preprocessing session, file and dataset exists, will not overwrite existing dataset.'.format(ID))

            else:  # conventional training session

                # Next step is to find out if the current processing session already exists. It is assumed that,
                # for conventional training, the session already exists if the following configuration file
                # elements are the same: the colour normalization file, the patch size, and the visibility level.
                # The comparison is achieved below.

                datasets = list(np.load(patch_npy_export_filename, allow_pickle=True))
                inner_keys = ['NORMALIZATION', 'DATA', 'DATA']
                outer_keys = ['Colour_Norm_File', 'Patch_Size', 'Vis']
                mask = np.full(len(datasets), True)
                for kk in range(len(inner_keys)):
                    datasets_vals = [dataset['header'][inner_keys[kk]][outer_keys[kk]] for dataset in datasets]
                    cur_dataset_vals = config[inner_keys[kk]][outer_keys[kk]]
                    mask = mask & [cur_dataset_vals == dataset_val for dataset_val in datasets_vals]

                formatted_indexes = []
                if any(mask):
                    formatted_indexes = list(np.where(mask)[0])

                if len(formatted_indexes) == 0:  # file does not match, so create it
                    WSI_processing_index.append(len(datasets))
                    processing_flag.append(True)
                    if verbose:
                        print('WSI {}.npy: conventional session, file exists, will append new dataset.'.format(ID))

                else:  # there is at least one file

                    # Is the file unique?
                    message = ''
                    if len(formatted_indexes) == 1:  # yes
                        WSI_processing_index.append(formatted_indexes[0])
                        if verbose:
                            print('WSI {}.npy: conventional session, file exists,'.format(ID), end='')
                            message = 'existing dataset'
                    else:  # no, then use the one with the best performance = the lowest validation loss
                        datasets_loss = [dataset['header']['PERFORMANCE']['val_loss'] for dataset in datasets]
                        best_index = np.argmin(datasets_loss)[0][0]
                        WSI_processing_index.append(best_index)
                        if verbose:
                            print('WSI {}.npy: conventional session, file exists,'.format(ID), end='')
                            message = 'existing dataset with lowest validation loss'

                    # Finally, does the file include pre-processing labels, or have you asked for no overwriting?
                    if ('tissue_type' in datasets[WSI_processing_index[-1]]['dataframe'].columns) or (not overwrite):
                        processing_flag.append(False)
                        if verbose:
                            print(" will use {} without re-processing.".format(message))
                    else:
                        processing_flag.append(True)
                        if verbose:
                            print(" will overwrite {}.".format(message))
    """



def LoadFileParameter(config, dataset):

    cur_basemodel_str = npyExportTools.basemodel_to_str(config)
    coords_file = pd.DataFrame()

    for svs_path in dataset.SVS_PATH:
        npy_path    = os.path.join(os.path.split(svs_path)[0], 'patches', os.path.split(svs_path)[1].replace('svs', 'npy'))
        coords_file = pd.concat([coords_file, np.load(npy_path, allow_pickle=True).item()[cur_basemodel_str][1]])

    return coords_file


def SaveFileParameter(config, df, column_to_add, label_to_add):

    df[label_to_add] = pd.Series(column_to_add, index=df.index)
    df = df.fillna(0)
    cur_basemodel_str = npyExportTools.basemodel_to_str(config)
    for svs_path, df_split in df.groupby(df.SVS_PATH):
        npy_path = os.path.join(os.path.split(svs_path)[0], 'patches', os.path.split(svs_path)[1].replace('svs', 'npy'))
        os.makedirs(os.path.split(npy_path)[0], exist_ok=True)  # in case folder is non-existent
        npy_dict = np.load(npy_path, allow_pickle=True).item() if os.path.exists(npy_path) else {}
        npy_dict[cur_basemodel_str] = [config, df_split.copy()]
        np.save(npy_path, npy_dict)


def QueryFromServer(config, **kwargs):
    print("Querying from Server")
    df = pd.DataFrame()
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
        df_criteria = pd.DataFrame([[row[0].val, Path(row[1].val).stem, row[2].val] for row in result], columns=["id_omero", "id_external", "Size"])                                  
        for nb, temp in enumerate(value): df_criteria[keys[nb]]   = temp

        ## Version 2 -- populate everything (fragile)
        """
        df_criteria = pd.DataFrame()            
        for row in result: ## Transform the results into a panda dataframe for each found match
            temp = pd.DataFrame([[row[0].val, row[1].val, row[2].val, *row[3].val.getMapValueAsMap().values()] for row in result],
                                columns=["id", "Name", "Size",*row[3].val.getMapValueAsMap().keys()]))            
            df_criteria = pd.concat([df_criteria, temp])                                    
        """
        df_criteria['SVS_PATH'] = [os.path.join(config['DATA']['SVS_Folder'], image_id+'.svs') for image_id in df_criteria['id_external']]
        
        df = pd.concat([df, df_criteria])
        
    conn.close()
    return df

def Synchronize(config, df):

    for index, image in df.iterrows():
        filepath = Path(config['DATA']['SVS_Folder'], image['id_external']+'.svs')  # Remove " [0]" at end of file

        if filepath.is_file():  # Exist
            if not filepath.stat().st_size == image['Size']:  # Corrupted
                print("File size doesn't match, redownloading")
                os.remove(filepath)
                download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])

        else:  ## Doesn't exist
            print("Doesn't exist")
            download_image(image['id_omero'], config['DATA']['SVS_Folder'], config['OMERO']['User'], config['OMERO']['Host'], config['OMERO']['Pw'])

            
