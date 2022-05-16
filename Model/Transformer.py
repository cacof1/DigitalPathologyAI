import torch
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from torch import nn
from torch import Tensor
from torch.nn.functional import softmax
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Model loosely based on
# - https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632,
# - https://amaarora.github.io/2021/01/18/ViT.html


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()

        # Reduce each of b images of size (c,h,w) into n = hw/(p^2) smaller patches of size (c,p,p).
        # Re-express each small patch as a vector with emb_size (e) elements: (c,p,p)->e.
        # This results in the mapping from (b,c,h,w) -> (b,n,e).

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # cls token of size (1,1,e)
        self.positions = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))  # pos embed, size (n+1, e)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])  # replicate cls token to size (b,1,e)
        x = torch.cat([cls_tokens, x], dim=1)  # prepend cls token to input
        x += self.positions  # add pos embed (will broadcast on b dimension)
        return x  # size (b, n+1, e)


class MSA(nn.Module): ## multihead self-attention
    def __init__(self, emb_size: int = 512, num_heads: int = 8, drop: float = 0):
        super().__init__()
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.MSA = nn.MultiheadAttention(emb_size, num_heads, dropout=drop, bias=False, batch_first=True)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Input arrays are of size (b, n+1, e), compatible with nn.MultiHeadAttention (with batch_first=True)
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        out, attention = self.MSA(queries, keys, values)  # TODO: export attention
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop: float = 0., forward_expansion: int = 4, ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                        nn.LayerNorm(emb_size),
                        MSA(emb_size, drop=drop, **kwargs),
                        )),
            ResidualAdd(nn.Sequential(
                        nn.LayerNorm(emb_size),
                        MLP(emb_size, expansion=forward_expansion, drop=drop),
                        ))
        )

class MLP(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop),
        )
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
                
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'), ## b is batch, n is number of sub-patches, e is embedding. Average over all sub-patches
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class ViT(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.in_channels = 3  # hard coded. This should not change, but you can add it to config file if needed.
        self.forward_features = nn.Sequential(
            PatchEmbedding(self.in_channels, config["DATA"]["Sub_Patch_Size_ViT"],
                           config["MODEL"]["Emb_Size_ViT"], config["DATA"]["Dim"][0][0]),
            TransformerEncoder(config["MODEL"]["Depth_ViT"], emb_size=config["MODEL"]["Emb_Size_ViT"],
                               num_heads=config['MODEL']['N_Heads_ViT'], drop=config['MODEL']['Drop_Rate'],
                               **kwargs),
        )

        self.classification_head = ClassificationHead(config["MODEL"]["Emb_Size_ViT"], config["DATA"]["N_Classes"])

        self.loss_fcn = getattr(torch.nn, self.config["MODEL"]["Loss_Function"])()

        if self.config['MODEL']['Loss_Function'] == 'CrossEntropyLoss':  # there is a bug currently. Quick fix...
            self.loss_fcn = torch.nn.CrossEntropyLoss(label_smoothing=self.config['REGULARIZATION']['Label_Smoothing'])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classification_head(x)  # classify using all heads
        #x = self.classification_head(torch.squeeze(x[:, 0, :]))  # classify only using learned cls token
        return x

    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        image         = next(iter(image.values())) ## Take the first value in the dictonnary for single zoom
        logits        = self.forward(image)
        loss          = self.loss_fcn(logits, labels)
        preds         = torch.argmax(softmax(logits, dim=1), dim=1)
        acc           = accuracy(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, labels = val_batch
        image         = next(iter(image.values())) ## Take the first value in the dictonnary for single zoom
        logits        = self.forward(image)
        loss          = self.loss_fcn(logits, labels)
        preds         = torch.argmax(softmax(logits, dim=1), dim=1)
        acc           = accuracy(preds, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def testing_step(self, test_batch, batch_idx):
        image, labels = test_batch
        image         = next(iter(image.values())) ## Take the first value in the dictonnary for single zoom
        logits        = self.forward(image)
        loss          = self.loss_fcn(logits, labels)
        preds         = torch.argmax(softmax(logits, dim=1), dim=1)
        acc           = accuracy(preds, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch
        image = next(iter(image.values())) ## Take the first value in the dictonnary for single zoom
        return softmax(self(image))

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['Algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['Weight_Decay'])

        if self.config['SCHEDULER']['Type'] == 'cosine_warmup':
            # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
            # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
            n_steps_per_epoch = self.config['DATA']['N_Training_Examples'] // self.config['MODEL']['Batch_Size']
            total_steps = n_steps_per_epoch * self.config['MODEL']['Max_Epochs']
            warmup_steps = self.config['SCHEDULER']['Cos_Warmup_Epochs'] * n_steps_per_epoch

            sched = transformers.optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=warmup_steps,
                                                                              num_training_steps=total_steps,
                                                                              num_cycles=0.5)  # default lr->0.

            scheduler = {'scheduler': sched,
                         'interval': 'step',
                         'frequency': 1}

        elif self.config['SCHEDULER']['Type'] == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.config["SCHEDULER"]["Lin_Step_Size"],
                                                        gamma=self.config["SCHEDULER"][
                                                            "Lin_Gamma"])  # step size 5, gamma =0.5

        return ([optimizer], [scheduler])
