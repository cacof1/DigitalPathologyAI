# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:05:03 2021

@author: zhuoy
"""

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.ops import box_iou
from utils.COCOengine import evaluate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def _evaluate_iou(target, pred):

    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

class FasterRCNN(pl.LightningModule):
    
    def __init__(self, dataset_train,dataset_test, num_classes=3, lr=0.005):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.train_dataset = dataset_train
        self.valid_dataset = dataset_test
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_filters = self.model.roi_heads.box_predictor.cls_score.in_features      
        self.model.roi_heads.box_predictor = FastRCNNPredictor(num_filters, num_classes)

    def forward(self, x, *args, **kwargs):
        return self.model(x)
        
        
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=1,
                                                   num_workers=0,
                                                   shuffle=True,
                                                   collate_fn=utils_.collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=1,
                                                   num_workers=0,
                                                   shuffle=False,
                                                   collate_fn=utils_.collate_fn)

        # prepare coco evaluator
#         coco = get_coco_api_from_dataset(valid_loader.dataset)
#         iou_types = _get_iou_types(self.model)
#         self.coco_evaluator = CocoEvaluator(coco, iou_types)

        return valid_loader
    
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
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()

        return {"val_iou": iou}

    def validation_epoch_end(self,valid_loader):
        
        res = {}
        APitems = ['IoU_0.50_0.95','IoU_0.50','IoU_0.75',
           'area_small','area_medium','area_large']
        ARitems = ['maxDets_1','maxDets_10','maxDets_100',
           'area_small','area_medium','area_large']
        
        coco_evaluator = evaluate(self.model, valid_loader, device=device)
        for i in range(6):            
            res["AP@{}".format(APitems[i])] = coco_evaluator.coco_eval['bbox'].stats[i]
            res["AR@{}".format(ARitems[i])] =  coco_evaluator.coco_eval['bbox'].stats[i+6]
            
        return res
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self):
        # return optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    
if __name__ == "__main__":

    filelist = sys.argv[1]
    
    data = (filelist, dataset_type = 'MitosisDetection')
    
    dataset = Dataset(data,train=True)
    dataset_test = Dataset(df_all,train=False)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = Subset(dataset, indices[:-100])
    dataset_test = Subset(dataset_test, indices[-100:])
    
    model = FasterRCNN(dataset_train,dataset_test)

    trainer = pl.Trainer(gpus=1, max_epochs=10)      
    trainer.fit(model)
