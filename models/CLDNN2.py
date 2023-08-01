from typing import List
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# https://github.com/brysef/rfml/blob/master/rfml/nn/model/cldnn.py
class CLDNN2(ModelBase):
    """Convolutional Long Deep Neural Network (CNN + GRU + MLP)

    This network is based off of a network for modulation classification first
    introduced in West/O'Shea.

    The following modifications/interpretations were made by Bryse Flowers <brysef@vt.edu>:

    - Batch Normalization was added otherwise the model was not stable enough to train
      in many cases (its unclear whether this was included in West's model)
    - The filter sizes were changed to 7 and the padding was set to 3 (where as
      West's paper said they used size 8 filters and did not mention padding)
        - An odd sized filter is necessary to ensure that the intermediate
          signal/feature map lengths are the same size and thus can be concatenated
          back together
    - A Gated Recurrent Unit (GRU) was used in place of a Long-Short Term Memory (LSTM).
        - These two submodules should behave nearly identically but GRU has one fewer
          equation
    - Bias was not used in the first convolution in order to more closely mimic the
      implementation of the CNN.
    - The hidden size of the GRU was set to be the number of classes it is trying to
      predict -- it makes the most sense instead of trying to find an arbritrary best
      hidden size.

    References
        N. E. West and T. O'Shea, “Deep architectures for modulation recognition,” in
        IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp.
        1-6, IEEE, 2017.
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
        num_classes = len(classes)
        self.example_input_array = torch.zeros((1,input_channels,input_samples), dtype=torch.cfloat)

        # Build model
        # Batch x IQ x input_samples

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2*input_channels, out_channels=50, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(50),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(50),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(50),
        )

        self.gru = nn.GRU(
            input_size=100,  # 100 channels after concatenation (50+50) * IQ (2)
            hidden_size=num_classes,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )

        self.mlp = nn.Sequential(
          # Flatten everything outside of batch dimension
          nn.Flatten(),

          # Fully connected layers
          # All of the outputs of the GRU are taken (instead of just the final hidden
          # output after all of the time samples).  Therefore, the number of "features"
          # after flattening is the time length * the hidden size * number of directions
          nn.Linear(input_samples * num_classes * 1, 256),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(256),
          nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = torch.view_as_real(x)
        x = x.transpose(-2,-1).flatten(1,2).contiguous()

        # Up front "filter" with no bias
        a = self.conv1(x)# Output is concatenated back as a "skip connection" below

        # Convolutional feature extraction layers
        x = self.conv2(a)
        
        # Concatenate the "skip connection" with the output of the rest of the CNN
        x = torch.cat((a, x), dim=1)

        # Flatten along channels and I/Q, preserving time dimension
        x = x.transpose(1, 2).contiguous() # BxCxT -> # BxTxC
        x = torch.flatten(x, 2) # BxTxCxIQ -> BxTxF

        # Temporal feature extraction
        x, _ = self.gru(x)

        # MLP Classification stage
        x = self.mlp(x)

        return x
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.00001)