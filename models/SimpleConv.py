from typing import List
import torchmetrics
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class SimpleConv(ModelBase):
    def __init__(
        self,
        classes: List[str],
        learning_rate: float = 0.0001,
    ):
        super().__init__(classes=classes)
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate

        # Build model
        self.model = nn.Sequential()
        for i in range(5):
            self.model.append(nn.Conv1d(2 if i==0 else 128, 128, 7))
            self.model.append(nn.BatchNorm1d(128))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Conv1d(128, 128, (5)))
            self.model.append(nn.BatchNorm1d(128))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.MaxPool1d(2))
        
        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(128*(22), 256)) #frame size after conv layers = 22
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Dropout())
        self.model.append(nn.Linear(256, 128))
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Dropout())
        self.model.append(nn.Linear(128, len(classes)))

        self.model = torch.compile(self.model)
        
    def forward(self, x):
        # x = torch.squeeze(x, 1)
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)