# 

import torch 
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channel, out_channel, kernel_size, stride):

    return nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride),
                         nn.BatchNorm3d(out_channel),
                         nn.ReLU()
                         )

class ConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        
        super().__init__()

        self.layers = [conv_block(in_channel, out_channel, kernel_size, stride) for in_channel, out_channel in zip(in_channels, out_channels)]

        self.layers.append(nn.Flatten())
    
        self.layers.append(nn.Dropout(0.1))

        self.layers.append(nn.LazyLinear(1))
                
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        
        return self.layers(x)


