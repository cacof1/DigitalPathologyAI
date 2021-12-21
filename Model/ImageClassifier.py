from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
import pytorch_lightning as pl
import sys
import torch
from torch.optim import Adam,AdamW
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision import datasets, models, transforms
from torch.nn.functional import softmax


class ImageClassifier(pl.LightningModule):

    def __init__(self, num_classes=2, lr=1e-3, weight_decay=0, backbone=models.densenet121(), lossfcn=nn.CrossEntropyLoss()):
        super().__init__()
        self.save_hyperparameters()  # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.backbone = backbone
        self.loss_fcn = lossfcn
        # self.model = nn.Sequential(self.backbone, nn.LazyLinear(512), nn.LazyLinear(num_classes)) # LazyLinear buggy
        out_feats = list(backbone.children())[-1].out_features
        # self.model = nn.Sequential(self.backbone, nn.Linear(out_feats, 512), nn.Linear(512, num_classes))
        self.model = nn.Sequential(self.backbone, nn.Linear(out_feats, num_classes))

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, labels = val_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def testing_step(self, test_batch, batch_idx):
        image, labels = test_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        preds = torch.argmax(softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch
        return softmax(self(image), dim=1)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer


if __name__ == "__main__":
    # Sample code to run a classifier - adjust to your needs.

    pl.seed_everything(42)

    MasterSheet = sys.argv[1]
    SVS_Folder = sys.argv[2]
    Patch_Folder = sys.argv[3]

    # For example, train on the first 1000 patches of each of the 10 first SFT.
    ids = WSIQuery(MasterSheet, diagnosis='solitary_fibrous_tumour', grade='low')[:10]
    wsi_file, coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)
    coords_file = coords_file[coords_file.index < 1000]

    transform = transforms.Compose([
        transforms.ToTensor(),  # this also normalizes to [0,1].
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data = DataModule(coords_file, wsi_file, train_transform=transform, val_transform=transform, batch_size=4,
                      inference=False, dim=(256, 256), target='sarcoma_label')

    model = ImageClassifier(backbone=models.densenet121(pretrained=True))
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=3)
    probabilities = trainer.fit(model, data)

    # Example to train from checkpoint :

    # checkpoint_callback = ModelCheckpoint(
    #    monitor='val_acc',
    #    dirpath=log_path,
    #    filename='{epoch:02d}-{val_acc:.2f}',
    #    save_top_k=1,
    # mode='max',
    # )

    # trainer = pl.Trainer(gpus=1, max_epochs=3,callbacks=[checkpoint_callback])
    # trainer.fit(model, data)
