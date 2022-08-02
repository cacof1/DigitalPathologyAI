# -*- coding: utf-8 -*-
"""
Created on Mon April 16 09:58:46 2022

@author: zhuoy
"""

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import transformers

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from pytorch_lightning.core.lightning import LightningModule
from Utils.ObjectDetectionTools import get_coco_api_from_dataset, MetricLogger, CocoEvaluator

#%%
class MaskFRCNN(LightningModule):
    def __init__(self, config, ):
        
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.num_classes = self.config["DATA"]["n_classes"]
        self.pre_trained = self.config['MODEL']['pretrained']
        self.APitems = ['IoU_0.50_0.95','IoU_0.50','IoU_0.75','area_small','area_medium','area_large']
        self.ARitems = ['maxDets_1','maxDets_10','maxDets_100','area_small','area_medium','area_large']
        
        if self.pre_trained:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,self.num_classes)
        else:
            backbone = resnet_fpn_backbone(self.config['MODEL']['backbone'], pretrained=True, trainable_layers=3)
            anchor_generator = AnchorGenerator(sizes=(16, 32,64,64,96), aspect_ratios=(0.75, 1.0, 1.35))
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=3, sampling_ratio=2)
            mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
            self.model = MaskRCNN(backbone, num_classes=self.num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,mask_roi_pool=mask_roi_pooler)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        
        if self.config['OPTIMIZER']['algorithm'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.config["OPTIMIZER"]["lr"],
                                        momentum=self.config['REGULARIZATION']['momentum'], 
                                        weight_decay=self.config['REGULARIZATION']['weight_decay'])
            
        else:
            optimizer = getattr(torch.optim, self.config['OPTIMIZER']['algorithm'])
            optimizer = optimizer(self.parameters(),
                                  lr=self.config["OPTIMIZER"]["lr"],
                                  eps=self.config["OPTIMIZER"]["eps"],
                                  betas=(0.9, 0.999),
                                  weight_decay=self.config['REGULARIZATION']['weight_decay'])

        if self.config['SCHEDULER']['type'] == 'cosine_warmup':
            batch_size = self.config['MODEL']['batch_size']
            max_epochs = self.config['MODEL']['max_epochs']
            n_steps_per_epoch = len(self.trainer._data_connector._train_dataloader_source.dataloader().dataset) // batch_size
            total_steps = n_steps_per_epoch * max_epochs
            warmup_steps = self.config['SCHEDULER']['warmup_epochs'] * n_steps_per_epoch

            sched = transformers.optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=warmup_steps,
                                                                              num_training_steps=total_steps,
                                                                              num_cycles=0.5)

            scheduler = {'scheduler': sched,
                         'interval': 'step',
                         'frequency': 1}
            
        elif self.config['SCHEDULER']['type'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.config["SCHEDULER"]["lin_step_size"],
                                                        gamma=self.config["SCHEDULER"]["lin_gamma"])

        return ([optimizer], [scheduler])

    def training_step(self, batch, batch_idx):

        images, targets = batch
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_classifier = loss_dict['loss_classifier']
        loss_box_reg = loss_dict['loss_box_reg']
        loss_objectness = loss_dict['loss_objectness']
        loss_mask = loss_dict['loss_mask']
        
        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_classifier', loss_classifier, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_loss_box_reg', loss_box_reg, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_loss_objectness', loss_objectness, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_mask', loss_mask, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return losses    
    
    def validation_step(self, batch, batch_idx):
 
        self.model.eval()
        images, targets = batch
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        losses = self.calculate_loss(images,targets)
        val_loss = sum(loss for loss in losses.values())

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return val_loss

    def validation_epoch_end(self, outputs):

        losses = sum(outputs)/self.config['MODEL']['num_of_gpus']
        coco_evaluator = self.evaluate(header='val')
        tensorboard_logs = {}
        for i in range(6):
            tensorboard_logs["AP@{}".format(self.APitems[i])] = coco_evaluator.coco_eval['segm'].stats[i]
            tensorboard_logs["AR@{}".format(self.ARitems[i])] = coco_evaluator.coco_eval['segm'].stats[i+6]
            
        self.log('val_AP', tensorboard_logs["AP@IoU_0.50"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_AP_all', tensorboard_logs["AP@IoU_0.50_0.95"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return losses
    
    def test_step(self, batch, batch_idx):
        self.model.eval()
        images, targets = batch
        images = list(image.cuda() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        losses = self.calculate_loss(images, targets)
        test_loss = sum(loss for loss in losses.values())

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return test_loss

    def test_epoch_end(self, outputs):

        losses = sum(outputs) / self.config['MODEL']['num_of_gpus']
        coco_evaluator = self.evaluate(header='test')
        tensorboard_logs = {}
        for i in range(6):
            tensorboard_logs["AP@{}".format(self.APitems[i])] = coco_evaluator.coco_eval['segm'].stats[i]
            tensorboard_logs["AR@{}".format(self.ARitems[i])] = coco_evaluator.coco_eval['segm'].stats[i+6]
            
        self.log('test_AP', tensorboard_logs["AP@IoU_0.50"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_AP_all', tensorboard_logs["AP@IoU_0.50_0.95"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return losses

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        images = batch
        images = list(image.cuda() for image in images)
        output = self.forward(images)

        return self.all_gather(output)

    @torch.no_grad()
    def evaluate(self, header):
        
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        
        if header == 'val':
            dataloader = self.trainer._data_connector._val_dataloader_source.dataloader()
        elif header == 'test':
            dataloader = self.trainer._data_connector._test_dataloader_source.dataloader()

        coco = get_coco_api_from_dataset(dataloader.dataset)
        iou_types = ['bbox','segm']
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(dataloader, 100, header):
            images = list(img.cuda() for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        
        return coco_evaluator

    def calculate_loss(self, images, targets):

        images, targets = self.model.transform(images, targets)
        features = self.model.backbone(images.tensors)

        self.model.rpn.training = True
        self.model.roi_heads.training = True

        proposals, proposal_losses = self.model.rpn(images, features, targets)
        detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)

        self.model.rpn.training = False
        self.model.roi_heads.training = False

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses
    
