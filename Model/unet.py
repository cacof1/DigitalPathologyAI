# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,depth, wf, in_channels, padding=True, batch_norm=True):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(depth):
            out_channels = 2**(wf+i)
            self.encoder.append(UNetDownBlock(in_channels, out_channels))
            in_channels = out_channels

        self.out_channels = in_channels
    def forward(self, x):
        for i, down in enumerate(self.encoder):
            x = down(x)
        return x

class Decoder(nn.Module):
    def __init__(self, depth, wf, in_channels, n_classes, padding=True, batch_norm=True):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 2**(wf+i)
            self.decoder.append(UNetUpBlock(in_channels, out_channels))
            in_channels  = out_channels            
        self.last = nn.Conv2d(out_channels, n_classes, kernel_size=1,stride=1)
        
    def forward(self,x):
        for i, up in enumerate(self.decoder):
            x = up(x)
        return self.last(x)            
                
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, depth=6, wf=6, padding=True, batch_norm=True):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
        """
        super(UNet, self).__init__()
        self.padding = padding
        self.depth = depth
        self.encoder = Encoder(depth, wf, in_channels)
        self.decoder = Decoder(depth, wf, self.encoder.out_channels, n_classes)

    ## Write the Modules
    def forward(self, x):
        x         = self.encoder(x)
        x         = self.decoder(x)
        return x
    
class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetDownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)),
            nn.PReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)),
            nn.BatchNorm2d(out_size),            
            nn.PReLU(),
            
            nn.Conv2d(out_size, out_size, kernel_size=2, padding=0,stride=2),
            nn.PReLU(),            
        )

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding="same"),
            nn.PReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_size) ,                       
            nn.PReLU(),
            )
        
    def forward(self, x):
        return self.block(x)
