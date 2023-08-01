from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# N. Soltani, K. Sankhe, S. Ioannidis, D. Jaisinghani, and K. Chowdhury, 
# “Spectrum Awareness at the Edge: Modulation Classification using Smartphones,” 
# 2019 IEEE International Symposium on Dynamic Spectrum Access Networks, DySPAN 2019, Nov. 2019, doi: 10.1109/DYSPAN.2019.8935775.
class DeepFIR(ModelBase):
    def __init__(
        self,
        input_samples: int,
        input_channels: int,
        classes: List[str],
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(classes=classes, **kwargs)
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)

        self.cconv = nn.Conv1d(input_channels, 128, 7, padding=3, dtype=torch.cfloat, bias=False)

        # Build model
        self.model = nn.Sequential()
        self.model.append(nn.BatchNorm1d(256))
        self.model.append(nn.ReLU(inplace=True))
        for i in range(5):
            self.model.append(nn.Conv1d(256 if i==0 else 128, 128, 7, padding=3))
            self.model.append(nn.BatchNorm1d(128))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Conv1d(128, 128, 5, padding=2))
            self.model.append(nn.BatchNorm1d(128))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.MaxPool1d(2))
        
        self.model.append(nn.Flatten())
        self.model.append(nn.LazyLinear(256))
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Dropout())
        self.model.append(nn.Linear(256, 128))
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Dropout())
        self.model.append(nn.Linear(128, len(classes)))

        # self.model = torch.compile(self.model)
        
    def forward(self, x):
        y = self.cconv(x)
        y = torch.view_as_real(y)
        y = y.transpose(-2,-1).flatten(1,2).contiguous()
        return self.model(y)
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.00001)