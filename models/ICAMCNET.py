from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        self.gen = torch.Generator().manual_seed(42)
        self.std = std
        self.mean = mean
        
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    

class CNN2(ModelBase):
    """

    References
        A. P. Hermawan, R. R. Ginanjar, D.-S. Kim, and J.-M. Lee, 
        “CNN-Based Automatic Modulation Classification for Beyond 5G Communications,” 
        IEEE Communications Letters, vol. 24, no. 5, pp. 1038-1041, May 2020, doi: 10.1109/LCOMM.2020.2970922.

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
        self.model.append(nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(8,1),
            padding='same',
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(4,1),
            padding='same',
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(8,1),
            padding='same',
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1,1),
            padding='same',
        ))
        self.model.append(nn.ReLU())
        self.model.append(nn.MaxPool2d((2,1)))
        self.model.append(nn.Dropout2d(0.4))

        # Flatten the input layer down to 1-d
        self.model.append(nn.Flatten())

        # Batch x Features
        self.model.append(nn.LazyLinear(128))
        self.model.append(nn.ReLU())
        self.model.append(nn.Dropout2d(0.4))
        self.model.append(AddGaussianNoise())
        self.model.append(nn.Linear(128, len(classes)))

    def forward(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        y = self.model(x)
        return y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00001)