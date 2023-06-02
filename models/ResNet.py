from typing import List
import torchmetrics
import torch
import torch.nn as nn
import math
from .ModelBase import ModelBase

class residual_unit(nn.Module):
    def __init__(self):
        super(residual_unit, self).__init__()

        self.conv0 = nn.Conv1d(32, 32, 3, padding='same')
        self.bn0 = nn.BatchNorm1d(32)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(32, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x += identity
        x = self.relu1(x)
        return x

class residual_stack(nn.Module):
    def __init__(self, input_chan):
        super(residual_stack, self).__init__()

        self.stack = nn.Sequential()
        self.stack.append(nn.Conv1d(input_chan, 32, 1, padding='same'))
        self.stack.append(residual_unit())
        self.stack.append(residual_unit())
        self.stack.append(nn.MaxPool1d(2, stride=2))

    def forward(self, x):
        return self.stack(x)
        

class ResNet(ModelBase):
    def __init__(
        self,
        classes: List[str],
        input_samples: int,
        learning_rate: float = 0.0001,
    ):
        super().__init__(classes=classes)

        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.example_input_array = torch.zeros((1,input_samples), dtype=torch.cfloat)

        # Build model
        self.model = nn.Sequential()

        self.model.append(residual_stack(2))
        self.model.append(residual_stack(32))
        self.model.append(residual_stack(32))
        self.model.append(residual_stack(32))
        self.model.append(residual_stack(32))
        self.model.append(residual_stack(32))
        
        frame_size_2 = 1024
        for i in range(6):
            frame_size_2 = math.floor((frame_size_2-2)/2+1)
        frame_size_2 = frame_size_2 * 32
            
        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(frame_size_2, 128))
        self.model.append(nn.SELU(inplace=True))
        # self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.AlphaDropout(0.3))
        self.model.append(nn.Linear(128, 128))
        self.model.append(nn.SELU(inplace=True))
        # self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.AlphaDropout(0.3))
        self.model.append(nn.Linear(128, len(classes)))
        
        
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)
