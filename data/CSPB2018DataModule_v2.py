import pytorch_lightning as pl
import h5py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from tqdm import tqdm
import scipy.constants as constants

class CSPB2018DataModule_v2(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size, frame_size: int = 1024, n_rx: int = 1):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.n_rx = n_rx

        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']

    def prepare_data(self): 
        pass

    def setup(self, stage: str = None):
        print('Preprocessing Data...')
        data = []
        with h5py.File(self.dataset_path, "r") as f:
            x = f['x']
            y = f['y'][()]
            t0 = f['T0']
            # cfo = f['cfo'][()]
            # beta = f['beta'][()]
            # upscale_factor = f['U'][()]
            # downscale_factor = f['D'][()]

            rng = np.random.default_rng()
            d = rng.uniform(1000, 10000, (x.shape[0], self.n_rx))
            snr = rng.uniform(0, 13, x.shape[0])
            cf = 900e6

            fsl = (4*np.pi*d*cf/constants.c)**2
            fsl = 10*np.log10(fsl)
            fsl_delta = fsl - np.min(fsl)
            snr = snr[:,None] - fsl_delta
            snr_linear = 10 ** (snr / 10)

            data = np.empty((x.shape[0], self.n_rx, x.shape[1]), dtype=np.complex64)
            for i in tqdm(range(len(x))):    
                noise = rng.standard_normal((self.n_rx, x.shape[1]*2), dtype=np.float32).view(np.complex64)
                noise = noise / np.sqrt(2)

                target_snr_linear = snr_linear[i]
                occupied_bw = 1 / t0[i]
                signal_power = np.mean(np.abs(x[i]) ** 2)
                signal_scale_linear = np.sqrt((target_snr_linear * occupied_bw) / signal_power)
                data[i] = x[i] * signal_scale_linear[:,None] + noise

        # Reshape data to frame_size
        # Need to be careful to use even multiples though so we don't break original fram barriers (32768 samples)
        if 32768 % self.frame_size:
            raise RuntimeError("frame_size is not an even divisor of the original frame size 32768.")
        data = data.reshape((-1, self.n_rx, self.frame_size))
        reshape_factor = 32768//self.frame_size
        y = np.repeat(y, reshape_factor)
        snr = np.repeat(snr, reshape_factor, 0)

        ds_full = TensorDataset(torch.from_numpy(data), torch.from_numpy(y).to(torch.long), torch.from_numpy(snr))

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
