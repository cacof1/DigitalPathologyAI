import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN,FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.ops import box_iou
from utils.COCOengine import evaluate
from Dataloader.DataloaderMitosis import DataModule_Mitosis
import segmentation_models_pytorch as smp
from utils import smp_functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MaskFRCNN(pl.LightningModule):
    
    def __init__(self, pre_trained = True, num_classes=2, lr=0.005):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.pre_trained = pre_trained
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
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
                
    def forward(self, image: torch.Tensor) -> torch.Tensor:  
        outputs = self.model(image)
        mask = outputs[0]['masks']
        return mask
        
    def training_step(self, batch):
        image, target = batch

        mask = target['masks']
                
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp_functional.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
                
    def validation_step(self, batch):
        
        image, target = batch
        mask = target['masks']
        
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp_functional.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def training_epoch_end(self, outputs):

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp_functional.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        dataset_iou = smp_functional.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            "train_per_image_iou": per_image_iou,
            "train_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)         


    def validation_epoch_end(self, outputs):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp_functional.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        dataset_iou = smp_functional.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            "valid_per_image_iou": per_image_iou,
            "valid_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)   

    def configure_optimizers(self):
        # return optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
