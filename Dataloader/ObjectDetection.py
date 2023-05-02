
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import openslide
from torchvision.transforms import functional as F
import nrrd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
#from skimage import morphology as morph

def get_bbox_from_mask(mask):
    pos = np.where(mask == 255)
    if pos[0].shape[0] == 0:
        return np.zeros((0, 4))
    else:
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target


transform = Compose([ToTensor(),
                     ])


class MFDataset(Dataset):

    def __init__(self,
                 df,
                 data_source='svs_files',
                 wsi_folder=None,
                 mask_folder=None,
                 nrrd_path=None,
                 augmentation=None,
                 normalization=None,
                 inference=False,
                 ):

        self.df = df
        self.wsi_folder = wsi_folder
        self.mask_folder = mask_folder
        self.nrrd_path = nrrd_path
        self.data_source = data_source
        self.transform = Compose([ToTensor(),
                                  ])

        self.augmentation = augmentation
        self.normalization = normalization
        self.inference = inference

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):

        vis_level = 0
        dim = (256, 256)
        if self.data_source == 'svs_files':

            index = self.df['index'][i]
            filename = self.df['SVS_ID'][i]
            top_left = (self.df['coords_x'][i], self.df['coords_y'][i])
            wsi_object = openslide.open_slide(self.wsi_folder + '{}.svs'.format(filename))
            img = np.array(wsi_object.read_region(top_left, vis_level, dim).convert("RGB"))

            num_objs = 1
            label = self.df['num_objs'][i]
            mask = np.load(self.mask_folder + '{}_masks.npy'.format(filename))[index]

        elif self.data_source == 'nrrd_files':
            custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}

            data, header = nrrd.read(os.path.join(self.nrrd_path, self.df['nrrd_file'][i]), custom_field_map)
            img = data[256:, 256:, :]
            mask = header['mask'][256:, 256:].astype('bool')
            #mask = morph.remove_small_objects(mask, min_size=300)
            mask = np.array(255 * mask)

            num_objs = 1

            if self.df['ann_label'][i] == 'yes':
                label = 1
            elif self.df['ann_label'][i] == 'no':
                label = 0

        if self.augmentation is not None:
            transformed = self.augmentation(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        masks = mask[np.newaxis, :, :]
        boxes = []
        area = []
        labels = []

        for n in range(num_objs):
            box = get_bbox_from_mask(masks[n])
            boxes.append(box)
            area.append((box[2] - box[0]) * (box[3] - box[1]))
            labels.append(label)

        obj_ids = np.array([255])
        masks = mask == obj_ids[:, None, None]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([i])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.normalization is not None:
            img = self.normalization(img)

        if self.inference:
            return img
        else:
            return img, target

class MixDataset(Dataset):

    def __init__(self,
                 df,
                 wsi_folder=None,
                 mask_folder=None,
                 masked_input=True,
                 data_source='nrrd_files',
                 dim=(64, 64),
                 vis_level=0,
                 channels=3,
                 nrrd_path=None,
                 transform=None,
                 extract_feature=False,
                 feature_setting=None,
                 inference=False):

        self.df = df
        self.wsi_folder = wsi_folder
        self.mask_folder = mask_folder
        self.masked_input = masked_input
        self.nrrd_path = nrrd_path
        self.data_source = data_source
        self.dim = dim
        self.vis_level = vis_level
        self.channels = channels
        self.extract_feature = extract_feature
        self.feature_setting = feature_setting
        self.transform = transform
        self.inference = inference

    def __getitem__(self, i):

        if self.data_source == 'svs_files':

            SVS_ID = self.df['SVS_ID'][i]
            top_left = (self.df['coords_x'][i], self.df['coords_y'][i])
            wsi_object = openslide.open_slide(self.wsi_folder + '{}.svs'.format(SVS_ID))
            img = np.array(wsi_object.read_region(top_left, self.vis_level, (256, 256)).convert("RGB"))
            box = [self.df['xmin'][i], self.df['ymin'][i], self.df['xmax'][i], self.df['ymax'][i]]

            if self.masked_input:
                index = self.df['index'][i]
                mask = np.load(self.mask_folder + '{}_detected_masks.npy'.format(SVS_ID))[index].astype('bool')
                box = get_bbox_from_mask(np.array(255 * mask))


        elif self.data_source == 'nrrd_files':
            custom_field_map = {
                'SVS_ID': 'string',
                'top_left': 'int list',
                'center': 'int list',
                'dim': 'int list',
                'vis_level': 'int',
                'diagnosis': 'string',
                'annotation_label': 'string',
                'mask': 'double matrix'}

            #print(self.df['nrrd_file'][i])

            data, header = nrrd.read(os.path.join(self.nrrd_path, self.df['nrrd_file'][i]), custom_field_map)
            img = data[256:, 256:, :]
            mask = header['mask'][256:, 256:].astype('bool')
            #mask = morph.remove_small_objects(mask, min_size=300)
            box = get_bbox_from_mask(np.array(255 * mask))

        center = ([(box[1] + box[3]) / 2, (box[0] + box[2]) / 2])
        center[0] = max(center[0], 32)
        center[0] = min(center[0], 224)
        center[1] = max(center[1], 32)
        center[1] = min(center[1], 224)

        if self.masked_input:
            mask_3d = np.stack((mask, mask, mask), axis=-1)
            masked_img = (img * mask_3d)[int(center[0] - 32):int(center[0] + 32), int(center[1] - 32):int(center[1] + 32), :]
            img = np.array(masked_img)
        else:
            img = img[int(center[0] - 32):int(center[0] + 32), int(center[1] - 32):int(center[1] + 32), :]
                
        if self.transform: img = self.transform(img)

        if self.inference:
            return img
        else:
            if self.data_source == 'svs_files':
                label = torch.as_tensor(self.df['gt_label'][i], dtype=torch.int64)
            elif self.data_source == 'nrrd_files':
                label = torch.as_tensor(self.df['ann_label'][i], dtype=torch.int64)

            return img, label

    def __len__(self):
        return self.df.shape[0]


class MFDataModule(LightningDataModule):
    def __init__(self,
                 df_train,
                 df_val,
                 df_test,
                 DataType='MFDataset',
                 batch_size=2,
                 num_of_worker=0,
                 augmentation=None,
                 normalization=None,
                 train_transform=None,
                 val_transform=None,
                 inference=False,
                 collate_fn=None,
                 **kwargs):

        super().__init__()
        self.batch_size = batch_size
        self.num_of_worker = num_of_worker
        self.DataType = DataType
        self.collate_fn = collate_fn

        if self.DataType == 'MFDataset':
            self.train_data = MFDataset(df_train, augmentation=augmentation, normalization=normalization, inference=inference, **kwargs)
            self.val_data = MFDataset(df_val, augmentation=None, normalization=normalization, inference=inference, **kwargs)
            self.test_data = MFDataset(df_test, augmentation=None, normalization=normalization, inference=inference, **kwargs)


        elif self.DataType == 'MixDataset':
            self.train_data = MixDataset(df_train,  transform=train_transform, **kwargs)
            self.val_data = MixDataset(df_val,  transform=val_transform, **kwargs)
            self.test_data = MixDataset(df_test, transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_of_worker, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_of_worker, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=self.num_of_worker, collate_fn=self.collate_fn)
