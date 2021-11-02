# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, depth=6, wf=6, padding=True, batch_norm=True):
        super(SimpleNet, self).__init__()        

        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 16, kernel_size=3, stride =1 ,padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride = 1, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3,stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding="same"),
            nn.LeakyReLU()            
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32,16, kernel_size=1,stride=1,padding="same"),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=1,stride=1,padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(32,64, kernel_size=1,stride=1,padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),


            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),            


            nn.Conv2d(64,32, kernel_size=3,stride=1,padding="same"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16, kernel_size=3,stride=1,padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(16,n_classes, kernel_size=3,stride=1,padding="same"),
            nn.LeakyReLU(),            
        )


        


    ## Write the Modules
    def forward(self, x):
        x         = self.encoder(x)
        x         = self.bottleneck(x)  
        x         = self.decoder(x)
        return x
