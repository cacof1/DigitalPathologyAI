import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from torch import nn
from torch import Tensor
from torch.nn.functional import softmax
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
class PatchEmbedding(nn.Module): ## split the images in sub-patches and preprend cls and position
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), ## 16^2*3 -- Learnable
            Rearrange('b e (h) (w) -> b (h w) e'), ## e -> embedding, h-># in height, w-> # in width, b-> batch, h*w -> n (total number of sub-patches)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size) ## Learnable
        self.queries = nn.Linear(emb_size, emb_size) ## Learnable
        self.values = nn.Linear(emb_size, emb_size) ## Learnable
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)  ## b is batch, n is total number of sub-patches, (hd) == e is embedding size which is split as into h (=num_heads) sub-embedding of size d
        keys    = rearrange(self.keys(x), "b n (h d) -> b h n d",    h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d",  h=self.num_heads)

        # sum up over the last axis -- tensor multiplication over the sub-embedding 
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # Size of batch, num_heads, n_subpatches, n_subpatches
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2) 
        att = F.softmax(energy, dim=-1) / scaling ## Attention scaling, dunno why
        att = self.att_drop(att)

        # ATT: batch, num_heads, n_subpatches, n_subpatches
        # VALUES: is batch, num_heads, n_subpatches, emb_size/num_heads,
        # OUT: batch, num_head, n_subpatches, emb_size/num_head -- matrix multiplication of the last two elements (al, lv -> av)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        
        out = rearrange(out, "b h n d -> b n (h d)") ## reverse the subsplitting --> batches, num_subpatches, emb_size
        att = reduce(att, "b h n c -> b n c", reduction="sum") ## add the attention from every sub embedding (batch, n_subpatches, n_subpatches)
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
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
        self.model = nn.Sequential(
            PatchEmbedding(config["DATA"]["n_channel"], config["DATA"]["sub_patch_size"], config["MODEL"]["emb_size"], config["DATA"]["dim"][0][0]),
            TransformerEncoder(config["MODEL"]["depth"], emb_size=config["MODEL"]["emb_size"], **kwargs),
            ClassificationHead(config["MODEL"]["emb_size"], config["DATA"]["n_classes"])
        )

        self.loss_fcn = getattr(torch.nn, self.config["MODEL"]["loss_function"])()
    def forward(self, x):
        return self.model(x)

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
        print(logits.dtype, labels.dtype)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["OPTIMIZER"]["lr"],
                                     eps=self.config["OPTIMIZER"]["eps"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["OPTIMIZER"]["step_size"],
                                                gamma=self.config["OPTIMIZER"]["gamma"])  
        return ([optimizer], [scheduler])
