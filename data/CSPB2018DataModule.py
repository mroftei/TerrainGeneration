import pytorch_lightning as pl
import glob
import os
import math
import sys
import hashlib
import urllib
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

CSPB2018_META = [
    ('https://cyclostationary.blog/wp-content/uploads/2019/02/batch_1.zip', 'd0e22ff42ffcbbd39046821a459b3329'), 
    ('https://cyclostationary.blog/wp-content/uploads/2019/02/batch_2.zip', 'ff2c303aba9af7bf4c70ce1390e1f445'), 
    ('https://cyclostationary.blog/wp-content/uploads/2019/02/batch_3.zip', 'ae2874ddabaac589897b610c86486bd4'), 
    ('https://cyclostationary.blog/wp-content/uploads/2019/02/batch_4.zip', '2c29f68f2070cb2509a09df55bc6abb3'), 
    ('https://cyclostationary.blog/wp-content/uploads/2019/02/batch_5.zip', '6054f4443f73423d39b09ccaceeb1e74'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_6.zip', 'f4e7641ecd940a10f4617ef5eb25e0bb'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_7.zip', '4a89de6d936369d492cfe9cfcb8d91ee'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_8.zip', '637a397ea36edb5cdffa09a8f6d65618'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_9.zip', 'd0fc525c836d4f5b4d9427904f60a532'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_10.zip', 'c224c27aee3a730824f10a45dfbdd8ac'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_11.zip', '7cb5781aa0b305a1fb877e685ac7be82'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_12.zip', '50733472d47f32db2cb66bf59194b003'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_13.zip', '4338c5195ed63cfd2d949b7540eacd42'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_14.zip', '14afc79e17e29ecc4b2878389cccafab'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_15.zip', '6f3f31ba3f70c10ff1deb2e56554c506'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_16.zip', '417c3dd056b0bc6365eea6c503a6a5b9'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_17.zip', 'da0232f80e84bef88c224e3e14eb3c95'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_18.zip', 'bffb83d36ff478d9cfadca3a2690c226'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_19.zip', '9e40d74866e5db88c96a0bde463fc22a'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_20.zip', '73030add98e7c413752c135c2cdc1492'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_21.zip', 'c91c2ce48880cab953b13379d8aa7839'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_22.zip', '8368e2a4aef68eefb160de97d7ba4708'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_23.zip', '9df4edae1a4cf0d730cba245df63b480'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_24.zip', '691968d41c705c856160064831dc3cf2'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_25.zip', '2268ba1c824a388568d60d890328e68c'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_26.zip', 'ab2f7bb4661dccef0f63998023c7a8f3'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_27.zip', '5bca81fe3ad3935d16fe8cc2b708f7a5'), 
    ('https://cyclostationary.blog/wp-content/uploads/2020/07/Batch_28.zip', '3ce6e5664b574e6fbf4c4954a6d3ae93'),
    ('https://cyclostationary.blog/wp-content/uploads/2020/09/signal_record_v2.txt', '764946ed2dade188f0f0837a64feef17'),
]

def check_file_integrity(fpath: str, md5: str) -> bool:
    if not os.path.isfile(fpath):
        return False
    
    if sys.version_info >= (3, 9):
        md5_alg = hashlib.md5(usedforsecurity=False)
    else:
        md5_alg = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            md5_alg.update(chunk)
    return md5 == md5_alg.hexdigest()

class CSPB2018DataModule(pl.LightningDataModule):
    def __init__(self, dataset_root: str, batch_size, frame_size: int = 1024, download: bool = False):
        super().__init__()
        self.dataset_root = os.path.expanduser(dataset_root)
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.download = download

        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']

    def prepare_data(self): 
        if self.download:
            extract_root = os.path.join(self.dataset_root, "extracted_signals")
            os.makedirs(extract_root, exist_ok=True)

            for url, zip_md5 in CSPB2018_META:
                fname = os.path.basename(url)
                if 'zip' in os.path.basename(url):
                    fpath = os.path.join(self.dataset_root, fname)
                else: 
                    fpath = os.path.join(extract_root, fname)

                # check if file is already present locally and download if not
                if check_file_integrity(fpath, zip_md5):
                    print("Using downloaded and verified file: " + fpath)
                else:
                    # download the file
                    print("Downloading " + url + " to " + fpath)
                    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pytorch/vision"})) as response:
                        content = iter(lambda: response.read(1024*32), b"")
                        with open(fpath, "wb") as fh:
                            for chunk in content:
                                # filter out keep-alive new chunks
                                if not chunk:
                                    continue
                                fh.write(chunk)

                # check integrity of downloaded file
                if not check_file_integrity(fpath, zip_md5):
                    raise RuntimeError("File not found or corrupted.")

                # Unzip zip file downloaded
                if 'zip' in fname:
                    print(f"Extracting {fpath} to {extract_root}")
                    with zipfile.ZipFile(fpath, "r", compression=zipfile.ZIP_STORED) as zipf:
                        zipf.extractall(extract_root)

    def setup(self, stage: str = None):
        print('Preprocessing Data...')
        fnames = glob.glob(os.path.join(self.dataset_root, "extracted_signals", "*", "signal*.tim"))
        fnames = sorted(fnames, key=lambda x:int(os.path.basename(x)[:-4].split('_')[-1])) 
        metadata_path = os.path.join(self.dataset_root, "extracted_signals", "signal_record_v2.txt")
        metadata_headers = ['label', 'T0', 'cfo', 'beta', 'U', 'D', 'SNR', 'P(N)']
        meatdata = pd.read_csv(metadata_path, header=None, delim_whitespace=True, index_col=0, names=metadata_headers)
        
        print("Loading data in to memory")
        x = torch.empty((112000, 1, 32768), dtype=torch.complex64)
        for i, fname in enumerate(fnames):
            x[i] = torch.from_numpy(np.fromfile(fname)[1:].view(np.complex64))
        y = torch.tensor([self.classes.index(l) for l in meatdata['label']], dtype=torch.long)
        snr = torch.from_numpy(meatdata['SNR'].to_numpy())

        # print("Normalizing...")
        # # Per-frame normalize to -1.0:1.0
        # # TODO: try 0:1 scaling
        # new_min, new_max = -1.0, 1.0
        # x_max = x.abs().amax(axis=(1,2), keepdim=True) # farthest value from 0 in each frame
        # scale = ((new_max - new_min) / (x_max*2))
        # x *= scale

        # scale randomly by 0.75-1.0
        # rng = np.random.default_rng(1024)
        # scale2 = rng.uniform(0.75, 1.0, y.shape[0])
        # x *= scale2[:,None,None]

        # Reshape data to frame_size
        # Need to be careful to use even multiples though so we don't break original fram barriers (32768 samples)
        if 32768 % self.frame_size:
            raise RuntimeError("frame_size is not an even divisor of the original frame size 32768.")
        x = x.reshape((-1, 1, self.frame_size))
        reshape_factor = 32768//self.frame_size
        y = np.repeat(y, reshape_factor)
        snr = np.repeat(snr, reshape_factor)

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
