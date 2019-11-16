# -*- coding: utf-8 -*
from torch import nn
import torch


class Net(nn.Module):
    def __init__(self, n_residual=8):
        super(Net, self).__init__()
        self.n_residual = n_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        for i in range(n_residual):
            self.add_module("ResidualBlock_"+str(i), ResidualBlock(64, 64))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.upsample = nn.Sequential(
            UpsampleBlock(64, 2),
            UpsampleBlock(64, 2),
            nn.Conv2d(64,3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        y = self.conv1(x)
        cache = y.clone()

        for i in range(self.n_residual):
            y = self.__getattr__("ResidualBlock_"+str(i))(y)

        y = self.conv2(y)
        y = self.upsample(y+cache)
        return (torch.tanh(y)+1.0)/2.0

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        return x+self.net(x)

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels, scaleFactor, k=3, p=1):
        super(UpsampleBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*(scaleFactor**2), kernel_size=k, padding=p),
            nn.PixelShuffle(scaleFactor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, l=0.2):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(l),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  ## [128+1*2-(3-1)-1]/2 +1 ->64^2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(l),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(l),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  ## -> 32^2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(l),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(l),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  ## -> 16^2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(l),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(l),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  ## -> 8^2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(l),

            nn.AdaptiveAvgPool2d(1),  ## [batch, 512, 1, 1]
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(l),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        y = self.net(x)
        # print ('D output size :' +  str(y.size()))
        si = torch.sigmoid(y).view(y.size()[0])  # [batch]
        # print ('D output : ' + str(si))
        return si