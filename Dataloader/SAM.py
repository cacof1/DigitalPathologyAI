import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn import preprocessing
import openslide
import torch
from torch.nn import functional as F
from collections import Counter
import itertools
# import Utils.sampling_schemes as sampling_schemes
from Utils.OmeroTools import *
from Utils import npyExportTools
from pathlib import Path
from QA.Normalization.Colour import ColourNorm
import nrrd
from segment_anything.utils.transforms import ResizeLongestSide

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, config, df, sam_img_size =1024, transform=None, inference=False):
        super().__init__()

        self.df = df
        self.sam_img_size  = sam_img_size
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        self.transform = transform
        self.inference = inference
        self.nrrd_folder = config['DATA']['nrrd_folder']
        self.custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        # load image
        nrrd_file = os.path.join(self.nrrd_folder, self.df['nrrd_file'].iloc[id])
        data, header = nrrd.read(nrrd_file, self.custom_field_map)

        input = {}

        img = data[256:, 256:, :]
        mask = header['mask'][256:, 256:].astype('bool')

        original_image_size = img.shape[:2]
        input_size = tuple(img.shape[-2:])

        input['image'] = img
        input['input_size'] = input_size
        input['original_image_size'] = original_image_size

        gt_mask_resized = torch.from_numpy(np.resize(mask, (1, 1, mask.shape[0], mask.shape[1])))
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        if self.inference:
            return input
        else:
            return input, gt_binary_mask


class MFDataModule(LightningDataModule):
    def __init__(self,
                 df_train,
                 df_val,
                 df_test,
                 batch_size=2,
                 num_of_worker=0,
                 train_transform=None,
                 val_transform=None,
                 **kwargs):

        super().__init__()
        self.batch_size = batch_size
        self.num_of_worker = num_of_worker

        self.train_data = DataGenerator(df_train,  transform=train_transform, **kwargs)
        self.val_data = DataGenerator(df_val,  transform=val_transform, **kwargs)
        self.test_data = DataGenerator(df_test, transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_of_worker)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_of_worker)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=self.num_of_worker)
