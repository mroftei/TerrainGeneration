from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# N. Soltani, K. Sankhe, S. Ioannidis, D. Jaisinghani, and K. Chowdhury, 
# “Spectrum Awareness at the Edge: Modulation Classification using Smartphones,” 
# 2019 IEEE International Symposium on Dynamic Spectrum Access Networks, DySPAN 2019, Nov. 2019, doi: 10.1109/DYSPAN.2019.8935775.
class CNNBlocks(ModelBase):
    def __init__(
        self,
        classes: List[str],
        input_samples: int,
        learning_rate: float = 0.0001,
        use_1d: bool = False,
    ):
        super().__init__(classes=classes)
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.use_1d = use_1d
        self.example_input_array = torch.zeros((1,1,input_samples), dtype=torch.cfloat)

        # Build model
        self.model = nn.Sequential()
        for i in range(5):
            if use_1d:
                self.model.append(nn.Conv1d(2 if i==0 else 128, 128, 7, padding=3))
                self.model.append(nn.BatchNorm1d(128))
                self.model.append(nn.ReLU(inplace=True))
                self.model.append(nn.Conv1d(128, 128, 5, padding=2))
                self.model.append(nn.BatchNorm1d(128))
                self.model.append(nn.ReLU(inplace=True))
                self.model.append(nn.MaxPool1d(2))
            else:
                self.model.append(nn.Conv2d(1 if i==0 else 128, 128, (7,1), padding=(3,0)))
                self.model.append(nn.BatchNorm2d(128))
                self.model.append(nn.ReLU(inplace=True))
                self.model.append(nn.Conv2d(128, 128, (5,1), padding=(2,0)))
                self.model.append(nn.BatchNorm2d(128))
                self.model.append(nn.ReLU(inplace=True))
                self.model.append(nn.MaxPool2d((2,1)))
        
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
        x = torch.view_as_real(x)
        if self.use_1d:
            x = x.transpose_(-2,-1).flatten(1,2).contiguous()
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)