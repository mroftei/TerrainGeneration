from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
class GRU2(ModelBase):
    """

    References
        D. Hong, Z. Zhang, and X. Xu, “Automatic modulation classification using recurrent neural networks,” 
        in 2017 3rd IEEE International Conference on Computer and Communications (ICCC), Dec. 2017, pp. 695--700. 
        doi: 10.1109/CompComm.2017.8322633.


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
        #GRU Unit
        self.gru = nn.GRU(input_size=2*input_channels, hidden_size=128, num_layers=2, batch_first=True)

        #DNN
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(len(classes))
        )

    def forward(self, x: torch.Tensor):
        x1 = torch.view_as_real(x)
        x1 = torch.transpose(x1, -2, -3)
        x1 = torch.flatten(x1, -2, -1)
        _, h_t = self.gru(x1)
        y = self.lin(h_t[-1])
        return y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)