# deep neural network predicting energy barriers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

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

class ml_model:

    def __init__(self, model_weight):
        
        # model settings
        in_channels = [1, 128, 256, 512] # [1, 128, 128]
        out_channels = in_channels[1 :] + [1024]
        kernel_size = 3
        stride = 1        
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.model = ConvNet(in_channels, out_channels, kernel_size, stride).to(self.device)
        # print(summary(self.model, (1, 9, 9, 9)))

        self.model.load_state_dict(torch.load(model_weight))
        self.model.eval() ### evaluate mode 

    def predict(self, local_neuron_map):
        
        local_neuron_map_tensor = torch.FloatTensor(local_neuron_map).unsqueeze(1)
        # print(local_neuron_map_tensor.shape)
        return self.model(local_neuron_map_tensor.to(self.device))
