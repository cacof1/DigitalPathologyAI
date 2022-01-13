from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from pytorch_lightning import LightningModule
def conv_bn(in_channels, out_channels, *args, **kwargs):
      return nn.Sequential(OrderedDict(
        {'conv': nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, *args, **kwargs),
         'bn': nn.BatchNorm2d(out_channels) }))

def conv_bn_up(in_channels, out_channels, stride=1,*args, **kwargs):
    output_padding = stride - 1
    return nn.Sequential(OrderedDict(
        {'conv': nn.ConvTranspose2d(in_channels, out_channels,kernel_size=3, padding=1, output_padding=output_padding, stride=stride,*args, **kwargs),
         'bn': nn.BatchNorm2d(out_channels) }))

def shortcut_cn(in_channels, expanded_channels, downsampling):
    return nn.Sequential(OrderedDict(
        {'conv' : nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=downsampling, bias=False),
         'bn' : nn.BatchNorm2d(expanded_channels)}))

def shortcut_cn_up(in_channels, expanded_channels, downsampling):
    output_padding = downsampling - 1
    return nn.Sequential(OrderedDict(
        {'conv' : nn.ConvTranspose2d(in_channels, expanded_channels, kernel_size=1, stride=downsampling, bias=False, output_padding=output_padding),
         'bn' : nn.BatchNorm2d(expanded_channels)}))


class ResNetResidualBlock(LightningModule): ## Basic residual block class for both and upsampling
    expansion = 1
    def __init__(self, in_channels, out_channels, conv = conv_bn, shortcut_fn = shortcut_cn, expansion=1, downsampling=1, activation=nn.ReLU, *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels        
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        ## Basic
        self.blocks = nn.Sequential(
            conv(self.in_channels, self.out_channels, bias=False, stride=self.downsampling),
            activation(),
            conv(self.out_channels, self.expanded_channels, bias=False),
        )
        
        """
        ## Bottleneck
        self.expansion = 4       
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
        """

        self.shortcut = shortcut_fn(self.in_channels,self.expanded_channels,self.downsampling)

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    

class ResNetLayer(LightningModule):
    def __init__(self, in_channels, out_channels, block=ResNetResidualBlock, n=1, *args, **kwargs):
        super().__init__()

        #We perform downsampling directly by convolutional layers that have a stride of 2.
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(LightningModule):
    """
    ResNet encoder composed by decreasing size with increasing features.
    """
    def __init__(self, in_channels=3, wf=5, depth =4, activation=nn.ReLU, block=ResNetResidualBlock, *args,**kwargs):
        super().__init__()
        self.out_channels = 0
        self.conv = conv_bn
        self.shortcut_fn = shortcut_cn
        self.blocks = nn.ModuleList()
        for i in range(depth):
            n = 2         
            self.out_channels = 2**(wf+i)
            self.blocks.append(ResNetLayer(in_channels * block.expansion,  self.out_channels, n=n, activation=activation, block=block, conv = self.conv, shortcut_fn = self.shortcut_fn, *args, **kwargs))
            in_channels = self.out_channels

    def forward(self, x):
        #x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResNetDecoder(LightningModule):
    """
    ResNet decoder composed by increasing size with decreasing features.
    """
    def __init__(self, in_channels=512, wf= 5, depth=4, n_classes =3, activation=nn.ReLU, block=ResNetResidualBlock, *args,**kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.conv = conv_bn_up
        self.shortcut_fn = shortcut_cn_up
        self.out_channels = 0
        for i in reversed(range(depth)):
            n = 2         
            self.out_channels = 2**(wf+i)
            self.blocks.append(ResNetLayer(in_channels * block.expansion,  self.out_channels, n=n, activation=activation, block=block, conv = self.conv, shortcut_fn = self.shortcut_fn, *args, **kwargs))
            in_channels = self.out_channels

        self.last    = nn.ConvTranspose2d(self.out_channels, n_classes, kernel_size=2,stride=2)        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x)
        x = self.last(x)
        return x

    
class ResNet(LightningModule):
    
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.out_channels,  *args, **kwargs)

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
