from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class DenseNet(ModelBase):
    """

    References
        X. Liu, D. Yang, and A. E. Gamal, “Deep neural network architectures for modulation classification,” 
        in 2017 51st Asilomar Conference on Signals, Systems, and Computers, Oct. 2017, pp. 915-919. 
        doi: 10.1109/ACSSC.2017.8335483.

    """

    def __init__(
        self,
        classes: List[str],
        input_samples: int,
        learning_rate: float = 0.0001,
    ):
        super().__init__(classes=classes)

        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        self.example_input_array = torch.zeros((1,1,input_samples), dtype=torch.cfloat)

        self.model = nn.Sequential()

        # Batch x 1-channel x input_samples x IQ 
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(3,1),
            padding='same',
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,2),
            padding='same',
        )
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=256+256,
            out_channels=80,
            kernel_size=(3,1),
            padding='same',
        )

        self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(
            in_channels=256+256+80,
            out_channels=80,
            kernel_size=(3,1),
            padding='same',
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.Dropout2d(0.6))
        self.model.append(nn.Flatten())

        # Batch x Features
        self.model.append(nn.LazyLinear(128))
        self.model.append(nn.ReLU())
        self.model.append(nn.Dropout2d(0.6))
        self.model.append(nn.Linear(128, len(classes)))

    def forward(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        y = self.conv1(x)
        y = self.relu1(y)

        y_1 = self.conv2(y)
        y = torch.cat((y, y_1), 1)

        y_1 = self.relu2(y)
        y_1 = self.conv3(y_1)
        y = torch.cat((y, y_1), 1)

        y = self.model(y)

        return y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00001)