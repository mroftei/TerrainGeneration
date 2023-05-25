from typing import List
import torchmetrics
import torch
import torch.nn as nn
from .ModelBase import ModelBase

# Modified https://github.com/qieaaa/Deep-Architectures-for-Modulation-Recognition/blob/master/cldnn/lstm.py
# https://ieeexplore.ieee.org/document/7920754
class RnnNet(ModelBase):
    def __init__(
        self,
        classes: List[str],
        learning_rate: float = 0.0001,
    ):
        super().__init__(classes=classes)
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate

        self.cnn = nn.Sequential()
        for i in range(1):
            self.cnn.append(nn.Conv1d(2 if i==0 else 128, 128, 7, padding='same'))
            self.cnn.append(nn.BatchNorm1d(128))
            self.cnn.append(nn.ReLU(inplace=True))
            self.cnn.append(nn.Conv1d(128, 128, 5, padding='same'))
            self.cnn.append(nn.BatchNorm1d(128))
            self.cnn.append(nn.ReLU(inplace=True))
            self.cnn.append(nn.MaxPool1d(2))
            self.cnn.append(nn.Flatten(2,-1))
        # self.rnn = nn.RNN(1024//2, 128, 2, nonlinearity='tanh', dropout=0.5, batch_first=True)
        self.rnn = nn.GRU(1024//2, 128, 2, dropout=0.5, batch_first=True)

        self.proj_head = nn.Sequential()
        self.proj_head.append(nn.Flatten())
        self.proj_head.append(nn.Linear(128*128, 256)) #frame size after conv layers = 22
        self.proj_head.append(nn.ReLU(inplace=True))
        self.proj_head.append(nn.Dropout())
        self.proj_head.append(nn.Linear(256, 128))
        self.proj_head.append(nn.ReLU(inplace=True))
        self.proj_head.append(nn.Dropout())
        self.proj_head.append(nn.Linear(128, len(classes)))


        # input_dim =Input(shape=input_sample,name = 'LSTM_architecture')
        # zero_pad_1 = ZeroPadding2D(padding =(0,2),data_format='channels_last')(input_dim)
        # conv_1 = Conv2D(64,(1,5),activation= 'relu',data_format='channels_last')(zero_pad_1)
        # drop_1 = Dropout(0.2)(conv_1)
        # zero_pad_2 = ZeroPadding2D((0,2),data_format='channels_last')(drop_1)
        # conv_2 = Conv2D(64,(1,5),activation= 'relu',data_format='channels_last')(zero_pad_2)
        # drop_2 = Dropout(0.2)(conv_2)
        # (None, 50, 2, 242)
        # merge = Concatenate(axis=2)([drop_1,drop_2])
        # merge_size = list(np.shape(merge))
        # _,concat_h,concat_w,units = np.shape(merge)
        # dimensions = int(concat_h)*int(concat_w)
        # units = int(units)
        # resh_model = Reshape((units,dimensions))(merge)
        # lstm = CuDNNLSTM(64)(resh_model)
        # fc_1 = Dense(128,activation='relu')(lstm)
        # out_layer = Dense(len(modulation),activation='softmax')(fc_1)

        # self.cnn1 = nn.Sequential()
        # self.cnn1.append(nn.Conv1d(2, 64, 5, padding='same'))
        # self.cnn1.append(nn.BatchNorm1d(64))
        # self.cnn1.append(nn.ReLU(inplace=True))
        # self.cnn1.append(nn.Dropout())
        # self.cnn2 = nn.Sequential()
        # self.cnn2.append(nn.Conv1d(64, 64, 5, padding='same'))
        # self.cnn2.append(nn.BatchNorm1d(64))
        # self.cnn2.append(nn.ReLU(inplace=True))
        # self.cnn2.append(nn.Dropout())
        # self.cnn2.append(nn.Conv1d(64, 64, 5, padding='same'))
        # self.cnn2.append(nn.BatchNorm1d(64))
        # self.cnn2.append(nn.ReLU(inplace=True))
        # self.cnn2.append(nn.Dropout())
        # self.flatten = nn.Flatten(2,-1)
        # self.rnn = nn.LSTM(1024*2, 64, 1, batch_first=True)
        # self.proj_head = nn.Sequential()
        # self.proj_head.append(nn.Flatten())
        # self.proj_head.append(nn.Linear(64*64, 128))
        # self.proj_head.append(nn.ReLU(inplace=True))
        # self.proj_head.append(nn.Dropout())
        # self.proj_head.append(nn.Linear(128, num_classes))
        
        
    def forward(self, x):
        # x = torch.squeeze(x, 1)
        x = self.cnn(x)
        # drop_1 = self.cnn1(x)
        # drop_2 = self.cnn2(drop_1)
        # x = torch.cat((drop_1, drop_2), -1)
        # x = self.flatten(x)
        x, _ = self.rnn(x)
        return self.proj_head(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00001)