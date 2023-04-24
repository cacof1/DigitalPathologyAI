import lightning as L
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision import models
from torch.nn.functional import softmax

## For the CM
from torchmetrics import ConfusionMatrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import io
from PIL import Image
from torchvision import models, transforms

class ConvNet(L.LightningModule):
    def __init__(self, config, label_encoder=None):
        super().__init__()

        self.save_hyperparameters()
        #self.validation_step_outputs = []
        #self.training_step_outputs = []
        self.config = config
        self.loss_fcn = nn.CrossEntropyLoss()#getattr(torch.nn, self.config["BASEMODEL"]["Loss_Function"])()
        self.LabelEncoder = label_encoder
        
        if self.config['BASEMODEL']['Loss_Function'] == 'CrossEntropyLoss':
            self.loss_fcn = torch.nn.CrossEntropyLoss(label_smoothing=self.config['REGULARIZATION']['Label_Smoothing'])

        self.activation = getattr(torch.nn, self.config["BASEMODEL"]["Activation"])()

        self.models = []
        for zoom_level in range(len(self.config['BASEMODEL']['Vis'])):
            backbone = getattr(models, config['BASEMODEL']['Backbone'])
            
            if 'densenet' in config['BASEMODEL']['Backbone']:
                backbone = backbone(weights='DEFAULT',
                                    drop_rate=config['ADVANCEDMODEL']['Drop_Rate'])
            else:
                backbone = backbone(weights='DEFAULT')

            out_feats = list(backbone.children())[-1].out_features

            self.models.append(backbone)
            self.add_module(f"model_{zoom_level}", self.models[zoom_level])

        self.classifier = nn.Sequential(
            nn.Linear(out_feats * len(config['BASEMODEL']['Vis']), 512),
            nn.Linear(512, self.config["DATA"]["N_Classes"]),
            self.activation,
        )

    def forward(self, x):
        aggregated_features = []
        for zoom_level in range(len(self.config['BASEMODEL']['Vis'])):
            features = self.models[zoom_level](x[:,zoom_level]) ## skip the batch
            aggregated_features.append(features)
        
        aggregated_features = torch.cat(aggregated_features, dim=1)
        return self.classifier(aggregated_features)

    def training_step(self, train_batch, batch_idx):
        image_dict, labels = train_batch
        logits = self.forward(image_dict)
        loss   = self.loss_fcn(logits, labels)        
        preds  = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.training_step_outputs.append({"loss": loss, "preds": preds, "labels": labels})                
        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_step(self, val_batch, batch_idx):
        image_dict, labels = val_batch
        logits = self.forward(image_dict)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.validation_step_outputs.append({"loss": loss, "preds": preds, "labels": labels})        
        return {"loss": loss, "preds": preds, "labels": labels}

    def test_step(self, test_batch, batch_idx):
        image_dict, labels = test_batch
        logits = self.forward(image_dict)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image_dict = batch
        output     = softmax(self(image_dict), dim=1)
        return self.all_gather(output)
    
    #def training_step_end(self, training_step_output):
    #    training_step_output = self.trainer.strategy.reduce([x['loss'] for x in self.training_step_output])
    #    self.training_step_outputs.append(training_step_output)
    #    return training_step_output

    #def validation_step_end(self, validation_step_output):
    #    self.validation_step_outputs.append(validation_step_output)

    #def on_train_epoch_end(self):
    #    epoch_average = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
    #    self.log("training_epoch_average", epoch_average,sync_dist=True)
    #    self.training_step_outputs.clear()  # free memory
        
    #def on_validation_epoch_end(self):
    #    #self.ConfusionMatrix()
    #    epoch_average = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
    #    self.log("validation_epoch_average", epoch_average,sync_dist=True)
    #    self.validation_step_outputs.clear()  # free memory    
        
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['Algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['Weight_Decay'])
    
        if self.config['SCHEDULER']['Type'] == 'cosine_warmup':
            n_steps_per_epoch = self.config['DATA']['N_Training_Examples'] // self.config['BASEMODEL']['Batch_Size']
            total_steps = n_steps_per_epoch * self.config['ADVANCEDMODEL']['Max_Epochs']
            warmup_steps = self.config['SCHEDULER']['Cos_Warmup_Epochs'] * n_steps_per_epoch
            
            sched = transformers.optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=warmup_steps,
                                                                              num_training_steps=total_steps,
                                                                            num_cycles=0.5)
            
            scheduler = {'scheduler': sched,
                         'interval': 'step',
                         'frequency': 1}
            
        elif self.config['SCHEDULER']['Type'] == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["SCHEDULER"]["Lin_Step_Size"], gamma=self.config["SCHEDULER"]["Lin_Gamma"])
        
        return ([optimizer], [scheduler])


    def ConfusionMatrix(self):
        tb = self.logger.experiment  # noqa
        outputs = torch.cat([tmp['preds'] for tmp in self.validation_step_outputs])
        labels  = torch.cat([tmp['labels'] for tmp in self.validation_step_outputs])
        le_name_mapping = dict(zip(self.LabelEncoder.classes_, self.LabelEncoder.transform(self.LabelEncoder.classes_)))

        confusion = ConfusionMatrix(num_classes=self.config["DATA"]["N_Classes"]).to(outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)
        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=le_name_mapping.values(),
            columns=le_name_mapping.values(),
        )

        # The heatmap assumes that the data is sorted - verify with
        fig, ax = plt.subplots(figsize=(17, 12))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.3)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax, xticklabels=le_name_mapping.values(), yticklabels=le_name_mapping.keys())
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

