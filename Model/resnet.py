from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

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

class ResNetResidualBlock(nn.Module): ## Basic residual block class for both and upsampling
    expansion = 1
    def __init__(self, in_channels, out_channels, conv, shortcut_fn, expansion=1, downsampling=1, activation=nn.ReLU, *args, **kwargs):
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
    

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetResidualBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by decreasing size with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256,512], deepths=[2,2,2,2],  activation=nn.ReLU, block=ResNetResidualBlock, *args,**kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,  out_channels, n=n, activation=activation, block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
                
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    ResNet decoder composed by increasing size with decreasing features.
    """
    def __init__(self, in_channels=512, blocks_sizes=[512,256,128,64,32,3], deepths=[2,2,2,2,2,2],  activation=nn.ReLU, block=ResNetResidualBlock, *args,**kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes        
        self.gate = nn.Sequential(
            nn.ConvTranspose2d(self.blocks_sizes[-1], 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
            nn.BatchNorm2d(3),
            activation(),
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,  out_channels, n=n, activation=activation, block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x)
        #x = self.gate(x)
        return x

    
class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, conv = conv_bn,   shortcut_fn = shortcut_cn, *args, **kwargs)
        self.decoder = ResnetDecoder(512, *args, conv = conv_bn_up, shortcut_fn = shortcut_cn_up, *args, **kwargs)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
