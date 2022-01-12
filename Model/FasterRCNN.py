# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:05:03 2021

@author: zhuoy
"""

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.ops import box_iou
from utils.COCOengine import evaluate
from Dataloader.DataloaderMitosis import DataModule_Mitosis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _evaluate_iou(target, pred):

    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

class FasterRCNN(pl.LightningModule):
    
    def __init__(self, pre_trained = True, num_classes=2, lr=0.005):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.pre_trained = pre_trained
        
        if self.pre_trained:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
        else:
            backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)

            anchor_generator = AnchorGenerator(sizes=(16, 32,64,128,256), aspect_ratios=(0.75, 1.0, 1.35))
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
            mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=14,sampling_ratio=2)
            self.model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,mask_roi_pool=mask_roi_pooler)
                
    def forward(self, x, *args, **kwargs):
        return self.model(x)
        
    
    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # separate losses
        loss_dict = self.model(images, targets)
        # total loss
        losses = sum(loss for loss in loss_dict.values())

        return {'loss': losses, 'log': loss_dict}
                #, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outputs)]).mean()
        #gt_boxes = [target['boxes'] for target in targets]
        #gt_boxes = list(chain(*gt_boxes))

        #pred_boxes = [output['boxes'] for output in outputs]
        #pred_boxes = list(chain(*pred_boxes))

        return {'val_iou':iou}       
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):
        # return optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
