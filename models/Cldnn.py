from typing import List
import torchmetrics
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cldnn.py
# N. E. West and T. O’Shea, “Deep architectures for modulation recognition,”
class Cldnn(ModelBase):
    def __init__(
        self,
        classes: List[str],
        learning_rate: float = 0.0001,
    ):
        super().__init__(classes=classes)
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate
        input_samples = 1024
        num_classes = len(classes)

        # Build model
        self.conv1 = nn.Sequential()
        self.conv2 = nn.Sequential()
        self.mlp = nn.Sequential()

        # Batch x IQ x input_samples
        self.conv1.append(nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, padding=3, bias=False))
        self.conv1.append(nn.ReLU(inplace=True))
        self.conv1.append(nn.BatchNorm1d(50))

        self.conv2.append(nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, padding=3, bias=True))
        self.conv2.append(nn.ReLU(inplace=True))
        self.conv2.append(nn.BatchNorm1d(50))
        self.conv2.append(nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, padding=3, bias=True))
        self.conv2.append(nn.ReLU(inplace=True))
        self.conv2.append(nn.BatchNorm1d(50))

        self.gru = nn.GRU(
            input_size=100,  # 100 channels after concatenation (50+50) * IQ (2)
            hidden_size=num_classes,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )

        # Flatten everything outside of batch dimension
        self.mlp.append(nn.Flatten())

        # Fully connected layers
        # All of the outputs of the GRU are taken (instead of just the final hidden
        # output after all of the time samples).  Therefore, the number of "features"
        # after flattening is the time length * the hidden size * number of directions
        self.mlp.append(nn.Linear(
            input_samples * num_classes * 1, 256
        ))
        self.mlp.append(nn.ReLU(inplace=True))
        self.mlp.append(nn.BatchNorm1d(256))

        self.mlp.append(nn.Linear(256, num_classes))

    def forward(self, x):
        # Up front "filter" with no bias
        a = self.conv1(x)# Output is concatenated back as a "skip connection" below

        # Convolutional feature extraction layers
        x = self.conv2(a)

        # Concatenate the "skip connection" with the output of the rest of the CNN
        # pylint: disable=no-member
        x = torch.cat((a, x), dim=1)

        # Flatten along channels and I/Q, preserving time dimension
        x = x.transpose(1, 2).contiguous() # BxCxT -> BxTxC

        # Temporal feature extraction
        x, _ = self.gru(x)

        # MLP Classification stage
        x = self.mlp(x)

        return x
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00001)