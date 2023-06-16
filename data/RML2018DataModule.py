import pytorch_lightning as pl
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

class RML2018DataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, batch_size, frame_size: int = 1024, use_hard: bool = True):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.use_hard = use_hard
        self.frame_size = 1024

        if use_hard:
            self.classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', 
                        '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', 
                        '32QAM', '64QAM', '128QAM', '256QAM', 
                        'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        else:
            self.classes = ['OOK', '4ASK', 'BPSK', 'QPSK', '8PSK',
                        '16QAM', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
            
        self.snrs = [-20, -18, -16, -14, -12, -10,  -8,  -6,  -4,  -2,   0,   2,   4,
            6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30]

    def prepare_data(self):            
        pass

    def setup(self, stage: str = None):
        print('Preprocessing Data...')
        f = h5py.File(self.data_file, 'r')

        x = f['X'][()]
        y = f['Y'][()]
        snr = f['Z'][()].flatten()

        # Remove unused classes
        to_remove = [17, 18] # AM-SSB-WC and AM-SSB-SC
        if not self.use_hard:
            to_remove = [2, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
        idx = y[:,to_remove].sum(1) == 0
        x = torch.as_tensor(x[idx])
        y = np.delete(y, to_remove, axis=1) # remove columns since onehot
        y = torch.as_tensor(y[idx], dtype=torch.int8)
        snr = torch.as_tensor(snr[idx])

        # Convery one-hot label to index
        y = torch.argmax(y, axis=1)

        # print("Normalizing...")
        # Per-frame normalize to -1.0:1.0
        # TODO: try 0:1 scaling
        # new_min, new_max = -1.0, 1.0
        # x_max = x.abs().amax(axis=(1,2), keepdim=True) # farthest value from 0 in each frame
        # scale = ((new_max - new_min) / (x_max*2))
        # x *= scale

        # scale randomly by 0.75-1.0
        # rng = np.random.default_rng(1024)
        # scale2 = rng.uniform(0.75, 1.0, y.shape[0])
        # x *= scale2[:,None,None]

        # Reshape data to have 2 channels matching PyTorch format (I,Q) (n,2,step_size)
        # can make frame and step size different here to generate more samples.
        # Need to be careful to use even multiples though so we don't break original fram barriers (1024 samples)
        # shape = (x.size//(self.frame_size*2), 2, int(self.frame_size))
        # strides = (x.strides[-1] * 2 * self.frame_size,
        #         x.strides[-1], x.strides[-1] * 2)
        # Apply sliding window to data
        # x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        # x = np.expand_dims(x, axis=1)
        # y = np.repeat(y, x.shape[0] / y.shape[0], axis=0)
        # snr = np.repeat(snr, x.shape[0] / snr.shape[0])

        x = torch.view_as_complex(x).unsqueeze(1)

        ds_full = TensorDataset(x, y, snr)

        self.ds_train, self.ds_val, self.ds_test = random_split(ds_full, [0.6, 0.2, 0.2], generator = torch.Generator().manual_seed(42))

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_val, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_test, shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True
        )
