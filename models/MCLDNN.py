from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class MCLDNN(ModelBase):
    """

    References
        J. Xu, C. Luo, G. Parr, and Y. Luo, 
        “A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition,” 
        IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1629-1632, Oct. 2020, 
        doi: 10.1109/LWC.2020.2999453.

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

        # Batch x 1-channel x input_samples x IQ 
        dr = 0.4  # dropout rate (%)
        #Cnvolutional Block

        # SeparateChannel Combined Convolutional Neural Networks
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 50, (8,2), padding='same'),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7,0), 0),
            nn.Conv1d(input_channels, 50, 8),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7,0), 0),
            nn.Conv1d(input_channels, 50, 8),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, (8,1), padding='same'),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, (5,2), padding='valid'),
            nn.ReLU()
        )
        #LSTM Unit
        self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, batch_first=True)

        #DNN
        self.lin = nn.Sequential(
            nn.Flatten(),
            # nn.LazyLinear(128),
            # nn.SELU(),
            # nn.Dropout(dr),
            # nn.LazyLinear(128),
            # nn.SELU(),
            # nn.Dropout(dr),
            nn.LazyLinear(len(classes))
        )

    def forward(self, x: torch.Tensor):
        x1 = torch.view_as_real(x)
        x1 = self.conv1(x1)

        x2 = self.conv2(x.real)
        x3 = self.conv3(x.imag)

        x2 = torch.unsqueeze(x2, -1) # B, 50, 1024, 1
        x3 = torch.unsqueeze(x3, -1)
        cat1 = torch.concatenate([x2,x3], dim=-1) # B, 50, 1024, 2
        x4 = self.conv4(cat1)

        cat2 = torch.concatenate([x4,x1], dim=-3) # B, 100, 1024, 2
        x5 = self.conv5(cat2)
        x5 = torch.squeeze(x5, -1)
        x5 = torch.transpose(x5, -1, -2)

        _, (h_t, _) = self.lstm(x5)

        y = self.lin(h_t[-1])
        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)