from typing import Any, List, Tuple
import pytorch_lightning as pl
import h5py
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.constants as constants

class MultiRxTransform:
    def __init__(self, n_rx):
        self.n_rx = n_rx
    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample

        return (x.repeat((self.n_rx,1)), y, t0)

class MultichannelNoiseTransform:
    """
    Multichannel noise, now independently distributed! FSPL calculations commented out
    """
    def __init__(self, min_snr, max_snr, generators: List[torch.Generator]):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.generators = generators

    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample
        noise = [torch.randn((1, x.shape[1]), dtype=torch.complex64, generator=gen) for gen in self.generators]
        noise = torch.cat(noise)

        # FSPL
        # d = torch.rand((self.n_rx))
        # d = (self.min_d - self.max_d) * d + self.max_d
        # fsl = (4*np.pi*d*self.cf/constants.c)**2
        # fsl = 10*torch.log10(fsl)
        # fsl_delta = fsl - torch.min(fsl)
        # snr = snr - fsl_delta

        snr = [torch.rand((1,), generator=gen) for gen in self.generators]
        snr = torch.cat(snr)
        snr = (self.min_snr - self.max_snr) * snr + self.max_snr # scale snr to min and max snr

        occupied_bw = 1 / t0
        signal_power = torch.mean(torch.abs(x) ** 2)
        target_snr_linear = 10 ** (snr / 10)
        signal_scale_linear = torch.sqrt((target_snr_linear * occupied_bw) / signal_power)
        x1 = x * signal_scale_linear[:,None] + noise
        return (x1, y, snr)

class TensorTransformDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors. Includes support for Transforms

    Each sample will be retrieved by indexing tensors along the first dimension. 
    Transforms will be passed the enire sample.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
        transforms (List of callable transforms): ordered list of transforms to perform
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, transforms=[]) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        sample = tuple(tensor[index] for tensor in self.tensors)
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __len__(self):
        return self.tensors[0].size(0)

class DisAMRDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size, frame_size: int = 1024, n_rx: int = 1, min_snr = 0, max_snr = 13, seed: int = 42):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.n_rx = n_rx
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.rng = torch.Generator().manual_seed(seed)

        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']

    def prepare_data(self): 
        pass

    def setup(self, stage: str = None):
        print('Preprocessing Data...')
        with h5py.File(self.dataset_path, "r") as f:
            x = torch.from_numpy(f['x'][()])
            y = torch.from_numpy(f['y'][()]).to(torch.long)
            t0 = torch.from_numpy(f['T0'][()])
            # snr = f['snr'][()]
            # cfo = f['cfo'][()]
            # beta = f['beta'][()]
            # upscale_factor = f['U'][()]
            # downscale_factor = f['D'][()]

        # Reshape data to frame_size
        # Need to be careful to use even multiples though so we don't break original fram barriers (32768 samples)
        if 32768 % self.frame_size:
            raise RuntimeError("frame_size is not an even divisor of the original frame size 32768.")
        x = x.reshape((-1, 1, self.frame_size))
        reshape_factor = 32768//self.frame_size
        y = torch.repeat_interleave(y, reshape_factor)
        t0 = torch.repeat_interleave(t0, reshape_factor)
        # snr = torch.repeat_interleave(snr, reshape_factor, 0)

        gens = [torch.Generator().manual_seed(self.rng.seed()+i) for i in range(self.n_rx)]

        transforms = [
            MultiRxTransform(self.n_rx),
            MultichannelNoiseTransform(self.min_snr, self.max_snr, gens)
        ]

        ds_full = TensorTransformDataset(x, y, t0, transforms=transforms)

        self.ds_train, self.ds_val, self.ds_test = random_split(ds_full, [0.6, 0.2, 0.2], generator = self.rng)

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
            persistent_workers=True,
            generator=self.rng
        )
