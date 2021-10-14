# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,depth, wf, in_channels, padding=True, batch_norm=True):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(UNetDownBlock(in_channels, 2**(wf+i), padding, batch_norm))
            in_channels = 2**(wf+i)
            
        self.out_channels = in_channels
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, depth, wf, in_channels, padding=True, batch_norm=True, up_mode='upconv', concat=True):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.decoder.append(UNetUpBlock(in_channels, 2**(wf+i), up_mode, padding, batch_norm , concat))
            in_channels = 2**(wf+i)        
        self.out_channels = in_channels            
    def forward(self,x):
        init = x
        for i, up in enumerate(self.decoder):
            x = up(x)
        return x


                
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=6, wf=6, padding=True,
                 batch_norm=True, up_mode='upconv' ,concat=True):
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
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.concat = concat
        self.encoder = Encoder(depth, wf, in_channels)
        self.decoder = Decoder(depth, wf, self.encoder.out_channels)
        self.last = nn.Conv2d(self.decoder.out_channels, n_classes, kernel_size=1)
        print(self.decoder.out_channels, n_classes)
    ## Write the Modules
    def forward(self, x):
        x         = self.encoder(x)
        x         = self.decoder(x)
        return self.last(x)
    
class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetDownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)),
            nn.ReLU(),
            nn.BatchNorm2d(out_size),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)),
            nn.ReLU(),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm , concat):
        super(UNetUpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=2,stride=2),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(out_size),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(out_size) ,           
            )
        
    def forward(self, x):
        return self.block(x)
