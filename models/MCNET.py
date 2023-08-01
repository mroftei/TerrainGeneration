from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class PreBlock(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=(7,3),
            stride=(2,1),
            padding=(3,1),
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((3,1), stride=(2,1), padding=(1,0))

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1),
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d((1,3), stride=(2,1), padding=(0,1))

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3,1),
            stride=(2,1),
            padding=(1,0),
            bias=True,
        )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        y1 = self.conv2(x)
        y1 = self.relu2(y1)
        y1 = self.pool2(y1)

        y2 = self.conv3(x)
        y2 = self.relu3(y2)

        return torch.cat((y1,y2), 1)

class MBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=True,
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1),
            bias=True,
        )
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3,1),
            stride=(1,1),
            padding=(1,0),
            bias=True,
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1,1),
            stride=(1,1),
            padding=(0,0),
            bias=True,
        )
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)

        y1 = self.conv2(x1)
        y1 = self.relu2(y1)

        y2 = self.conv3(x1)
        y2 = self.relu3(y2)

        y3 = self.conv4(x1)
        y3 = self.relu4(y3)

        return torch.cat((y1,y2,y3), 1)

class MBlockP(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=True,
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(1,3),
            stride=(1,1),
            padding=(0,1),
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d((3,1), stride=(2,1), padding=(1,0))

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3,1),
            stride=(2,1),
            padding=(1,0),
            bias=True,
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1,1),
            stride=(2,1),
            padding=(0,0),
            bias=True,
        )
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)

        y1 = self.conv2(x1)
        y1 = self.relu2(y1)
        y1 = self.pool(y1)

        y2 = self.conv3(x1)
        y2 = self.relu3(y2)

        y3 = self.conv4(x1)
        y3 = self.relu4(y3)

        return torch.cat((y1,y2,y3), 1)


# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class MCNET(ModelBase):
    """

    References
        T. Huynh-The, C.-H. Hua, Q.-V. Pham, and D.-S. Kim, 
        “MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification,” 
        IEEE Communications Letters, vol. 24, no. 4, pp. 811-815, Apr. 2020, doi: 10.1109/LCOMM.2020.2968030.

    """

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

        self.preblock = PreBlock(input_channels)

        self.skip1conv = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(2,1), padding=(0,0), bias=True)
        self.skip1pool = nn.MaxPool2d((3,1), stride=(2,1), padding=(1,0))
        self.pool1 = nn.MaxPool2d((3,1), (2,1), (1,0))
        self.Mblockp1 = MBlockP(64)

        self.Mblock2 = MBlock(128)

        self.skip3pool = nn.MaxPool2d((3,1), stride=(2,1), padding=(1,0))
        self.Mblockp3 = MBlockP(128)

        self.Mblock4 = MBlock(128)

        self.skip5pool = nn.MaxPool2d((3,1), stride=(2,1), padding=(1,0))
        self.Mblockp5 = MBlockP(128)

        self.Mblock6 = MBlock(128)

        self.poolavg = nn.AvgPool2d((8,2))
        self.dropout = nn.Dropout2d(0.5)
        self.linear = nn.LazyLinear(len(classes))

    def forward(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        y = self.preblock(x)

        y_1 = self.skip1conv(y)
        y_1 = self.skip1pool(y_1)
        y_2 = self.pool1(y)
        y_2 = self.Mblockp1(y_2)
        y = y_1 + y_2

        y_2 = self.Mblock2(y)
        y = y + y_2

        y_1 = self.skip3pool(y)
        y_2 = self.Mblockp3(y)
        y = y_1 + y_2

        y_2 = self.Mblock4(y)
        y = y + y_2

        y_1 = self.skip5pool(y)
        y_2 = self.Mblockp5(y)
        y = y_1 + y_2

        y_2 = self.Mblock6(y)
        y = torch.cat((y, y_2), 1)

        y = self.poolavg(y)
        y = self.dropout(y)
        y = y.flatten(1)
        y = self.linear(y)

        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)