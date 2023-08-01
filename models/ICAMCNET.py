from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean
        
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.get_device()) * self.std + self.mean
    

class ICAMCNET(ModelBase):
    """

    References
        A. P. Hermawan, R. R. Ginanjar, D.-S. Kim, and J.-M. Lee, 
        “CNN-Based Automatic Modulation Classification for Beyond 5G Communications,” 
        IEEE Communications Letters, vol. 24, no. 5, pp. 1038-1041, May 2020, doi: 10.1109/LCOMM.2020.2970922.

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
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=(7,1),
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,1),
                padding='same',
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(7,1),
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d((1 ,1)),
            nn.Dropout2d(0.4),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(7,1),
                padding='same',
            ),
            nn.ReLU(),
            nn.Dropout2d(0.4),

            # Flatten the input layer down to 1-d
            nn.Flatten(),

            # Batch x Features
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout1d(0.4),
            AddGaussianNoise(),
            nn.Linear(128, len(classes)),
        )

    def forward(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        y = self.model(x)
        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)