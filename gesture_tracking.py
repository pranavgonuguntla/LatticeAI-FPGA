import torch
import torch.nn as nn

class GestureCNN(nn.Module):  
    def __init__(self, num_classes=6): # Num_classes depends on the number of different gestures in the dataset (6 is placeholder)
        super(GestureCNN, self).__init__()  
        
        # Convolutional layers
        self.conv_one = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_two = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_three = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)


