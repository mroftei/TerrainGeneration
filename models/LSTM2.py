from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class LSTM2(ModelBase):
    """

    References
        S. Rajendran, W. Meert, D. Giustiniano, V. Lenders, and S. Pollin, 
        “Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors,” 
        IEEE Transactions on Cognitive Communications and Networking, vol. 4, no. 3, pp. 433-445, Sep. 2018, 
        doi: 10.1109/TCCN.2018.2835460.


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
        #LSTM Unit
        self.lstm = nn.LSTM(input_size=2*input_channels, hidden_size=128, num_layers=2, batch_first=True)

        #DNN
        self.lin = nn.Sequential(
            nn.Flatten(),
            # nn.LazyLinear(128),
            # nn.ReLU(),
            # nn.LazyLinear(128),
            # nn.ReLU(),
            nn.LazyLinear(len(classes))
        )

    def forward(self, x: torch.Tensor):
        x1 = torch.view_as_real(x)
        x1 = torch.transpose(x1, -2, -3)
        x1 = torch.flatten(x1, -2, -1)
        _, (h_t, _) = self.lstm(x1)
        y = self.lin(h_t[-1])
        return y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00001)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    
        schedulers = []
        # schedulers.append({
        #     "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1),
        # })
        # schedulers.append({
        #     "scheduler": torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/10, total_iters=1000),
        #     "interval": "step"
        # })

        # scheduler1 = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, threshold=1e-3, min_lr=1e-6),
        #     "interval": "epoch",
        #     "frequency": 1,
        #     "monitor": "val/F1",
        #     "reduce_on_plateau": True,
        # }
        return [optimizer], schedulers