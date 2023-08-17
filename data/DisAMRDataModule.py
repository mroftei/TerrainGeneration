from typing import Any, List, Tuple, Optional
import pytorch_lightning as pl
import h5py
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import scipy.constants as constants

class MultiRxTransform:
    def __init__(self, n_rx):
        self.n_rx = n_rx
    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample
        return (torch.repeat_interleave(x, 6, -2), y, t0)

class MultichannelNoiseTransform:
    """
    Multichannel noise, now independently distributed! FSPL calculations commented out
    """
    def __init__(self, min_snr, max_snr, generator: Optional[torch.Generator]=None):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.generator = generator

    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample
        # noise = [torch.randn((1, x.shape[1]), dtype=torch.complex64, generator=gen) for gen in self.generators]
        # noise = torch.cat(noise)
        noise = torch.randn(x.shape, dtype=torch.complex64, generator=self.generator, device=x.device) # Pregenerate unity gain noise

        # FSPL
        # d = torch.rand((self.n_rx))
        # d = (self.min_d - self.max_d) * d + self.max_d
        # fsl = (4*np.pi*d*self.cf/constants.c)**2
        # fsl = 10*torch.log10(fsl)
        # fsl_delta = fsl - torch.min(fsl)
        # snr = snr - fsl_delta

        # Randomly select SNR for each channel
        # snr = [torch.rand((1,), generator=gen) for gen in self.generators]
        # snr = torch.cat(snr)
        snr = torch.rand((x.shape[:-1]), generator=self.generator, device=x.device)
        snr = (self.min_snr - self.max_snr) * snr + self.max_snr # scale snr to min and max snr

        # Calculate power and scaling factor to match desired SNR
        occupied_bw = 1 / t0[:, None]
        signal_power = torch.mean(torch.abs(x) ** 2, -1)
        target_snr_linear = torch.pow(10, (snr / 10))
        signal_scale_linear = torch.sqrt((target_snr_linear * occupied_bw) / signal_power)
        x1 = x * signal_scale_linear[...,None] + noise
        return (x1, y, snr)

class SROTransform:
    def __init__(self, sample_rate_hz: float, std_dev_walk_hz: float, std_dev_init_hz:float, initial_offset: bool = False, generator: Optional[torch.Generator]=None):
        self.samp_rate = sample_rate_hz
        self.std_dev_init_hz = std_dev_init_hz
        self.std_dev_walk_hz = std_dev_walk_hz
        self.initial_offset = initial_offset
        self.generator = generator
        self.taps = torch.fliplr(torch.tensor([ # Interpolating filter taps
            #    -4            -3          -2          -1          0            1           2            3            mu
            [ 0.00000e+00,0.00000e+00, 0.00000e+00,0.00000e+00,1.00000e+00, 0.00000e+00,0.00000e+00, 0.00000e+00], #   0/128
            [-1.54700e-04,8.53777e-04,-2.76968e-03,7.89295e-03,9.98534e-01,-5.41054e-03,1.24642e-03,-1.98993e-04], #   1/128
            [-3.09412e-04,1.70888e-03,-5.55134e-03,1.58840e-02,9.96891e-01,-1.07209e-02,2.47942e-03,-3.96391e-04], #   2/128
            [-4.64053e-04,2.56486e-03,-8.34364e-03,2.39714e-02,9.95074e-01,-1.59305e-02,3.69852e-03,-5.92100e-04], #   3/128
            [-6.18544e-04,3.42130e-03,-1.11453e-02,3.21531e-02,9.93082e-01,-2.10389e-02,4.90322e-03,-7.86031e-04], #   4/128
            [-7.72802e-04,4.27773e-03,-1.39548e-02,4.04274e-02,9.90917e-01,-2.60456e-02,6.09305e-03,-9.78093e-04], #   5/128
            [-9.26747e-04,5.13372e-03,-1.67710e-02,4.87921e-02,9.88580e-01,-3.09503e-02,7.26755e-03,-1.16820e-03], #   6/128
            [-1.08030e-03,5.98883e-03,-1.95925e-02,5.72454e-02,9.86071e-01,-3.57525e-02,8.42626e-03,-1.35627e-03], #   7/128
            [-1.23337e-03,6.84261e-03,-2.24178e-02,6.57852e-02,9.83392e-01,-4.04519e-02,9.56876e-03,-1.54221e-03], #   8/128
            [-1.38589e-03,7.69462e-03,-2.52457e-02,7.44095e-02,9.80543e-01,-4.50483e-02,1.06946e-02,-1.72594e-03], #   9/128
            [-1.53777e-03,8.54441e-03,-2.80746e-02,8.31162e-02,9.77526e-01,-4.95412e-02,1.18034e-02,-1.90738e-03], #  10/128
            [-1.68894e-03,9.39154e-03,-3.09033e-02,9.19033e-02,9.74342e-01,-5.39305e-02,1.28947e-02,-2.08645e-03], #  11/128
            [-1.83931e-03,1.02356e-02,-3.37303e-02,1.00769e-01,9.70992e-01,-5.82159e-02,1.39681e-02,-2.26307e-03], #  12/128
            [-1.98880e-03,1.10760e-02,-3.65541e-02,1.09710e-01,9.67477e-01,-6.23972e-02,1.50233e-02,-2.43718e-03], #  13/128
            [-2.13733e-03,1.19125e-02,-3.93735e-02,1.18725e-01,9.63798e-01,-6.64743e-02,1.60599e-02,-2.60868e-03], #  14/128
            [-2.28483e-03,1.27445e-02,-4.21869e-02,1.27812e-01,9.59958e-01,-7.04471e-02,1.70776e-02,-2.77751e-03], #  15/128
            [-2.43121e-03,1.35716e-02,-4.49929e-02,1.36968e-01,9.55956e-01,-7.43154e-02,1.80759e-02,-2.94361e-03], #  16/128
            [-2.57640e-03,1.43934e-02,-4.77900e-02,1.46192e-01,9.51795e-01,-7.80792e-02,1.90545e-02,-3.10689e-03], #  17/128
            [-2.72032e-03,1.52095e-02,-5.05770e-02,1.55480e-01,9.47477e-01,-8.17385e-02,2.00132e-02,-3.26730e-03], #  18/128
            [-2.86289e-03,1.60193e-02,-5.33522e-02,1.64831e-01,9.43001e-01,-8.52933e-02,2.09516e-02,-3.42477e-03], #  19/128
            [-3.00403e-03,1.68225e-02,-5.61142e-02,1.74242e-01,9.38371e-01,-8.87435e-02,2.18695e-02,-3.57923e-03], #  20/128
            [-3.14367e-03,1.76185e-02,-5.88617e-02,1.83711e-01,9.33586e-01,-9.20893e-02,2.27664e-02,-3.73062e-03], #  21/128
            [-3.28174e-03,1.84071e-02,-6.15931e-02,1.93236e-01,9.28650e-01,-9.53307e-02,2.36423e-02,-3.87888e-03], #  22/128
            [-3.41815e-03,1.91877e-02,-6.43069e-02,2.02814e-01,9.23564e-01,-9.84679e-02,2.44967e-02,-4.02397e-03], #  23/128
            [-3.55283e-03,1.99599e-02,-6.70018e-02,2.12443e-01,9.18329e-01,-1.01501e-01,2.53295e-02,-4.16581e-03], #  24/128
            [-3.68570e-03,2.07233e-02,-6.96762e-02,2.22120e-01,9.12947e-01,-1.04430e-01,2.61404e-02,-4.30435e-03], #  25/128
            [-3.81671e-03,2.14774e-02,-7.23286e-02,2.31843e-01,9.07420e-01,-1.07256e-01,2.69293e-02,-4.43955e-03], #  26/128
            [-3.94576e-03,2.22218e-02,-7.49577e-02,2.41609e-01,9.01749e-01,-1.09978e-01,2.76957e-02,-4.57135e-03], #  27/128
            [-4.07279e-03,2.29562e-02,-7.75620e-02,2.51417e-01,8.95936e-01,-1.12597e-01,2.84397e-02,-4.69970e-03], #  28/128
            [-4.19774e-03,2.36801e-02,-8.01399e-02,2.61263e-01,8.89984e-01,-1.15113e-01,2.91609e-02,-4.82456e-03], #  29/128
            [-4.32052e-03,2.43930e-02,-8.26900e-02,2.71144e-01,8.83893e-01,-1.17526e-01,2.98593e-02,-4.94589e-03], #  30/128
            [-4.44107e-03,2.50946e-02,-8.52109e-02,2.81060e-01,8.77666e-01,-1.19837e-01,3.05345e-02,-5.06363e-03], #  31/128
            [-4.55932e-03,2.57844e-02,-8.77011e-02,2.91006e-01,8.71305e-01,-1.22047e-01,3.11866e-02,-5.17776e-03], #  32/128
            [-4.67520e-03,2.64621e-02,-9.01591e-02,3.00980e-01,8.64812e-01,-1.24154e-01,3.18153e-02,-5.28823e-03], #  33/128
            [-4.78866e-03,2.71272e-02,-9.25834e-02,3.10980e-01,8.58189e-01,-1.26161e-01,3.24205e-02,-5.39500e-03], #  34/128
            [-4.89961e-03,2.77794e-02,-9.49727e-02,3.21004e-01,8.51437e-01,-1.28068e-01,3.30021e-02,-5.49804e-03], #  35/128
            [-5.00800e-03,2.84182e-02,-9.73254e-02,3.31048e-01,8.44559e-01,-1.29874e-01,3.35600e-02,-5.59731e-03], #  36/128
            [-5.11376e-03,2.90433e-02,-9.96402e-02,3.41109e-01,8.37557e-01,-1.31581e-01,3.40940e-02,-5.69280e-03], #  37/128
            [-5.21683e-03,2.96543e-02,-1.01915e-01,3.51186e-01,8.30432e-01,-1.33189e-01,3.46042e-02,-5.78446e-03], #  38/128
            [-5.31716e-03,3.02507e-02,-1.04150e-01,3.61276e-01,8.23188e-01,-1.34699e-01,3.50903e-02,-5.87227e-03], #  39/128
            [-5.41467e-03,3.08323e-02,-1.06342e-01,3.71376e-01,8.15826e-01,-1.36111e-01,3.55525e-02,-5.95620e-03], #  40/128
            [-5.50931e-03,3.13987e-02,-1.08490e-01,3.81484e-01,8.08348e-01,-1.37426e-01,3.59905e-02,-6.03624e-03], #  41/128
            [-5.60103e-03,3.19495e-02,-1.10593e-01,3.91596e-01,8.00757e-01,-1.38644e-01,3.64044e-02,-6.11236e-03], #  42/128
            [-5.68976e-03,3.24843e-02,-1.12650e-01,4.01710e-01,7.93055e-01,-1.39767e-01,3.67941e-02,-6.18454e-03], #  43/128
            [-5.77544e-03,3.30027e-02,-1.14659e-01,4.11823e-01,7.85244e-01,-1.40794e-01,3.71596e-02,-6.25277e-03], #  44/128
            [-5.85804e-03,3.35046e-02,-1.16618e-01,4.21934e-01,7.77327e-01,-1.41727e-01,3.75010e-02,-6.31703e-03], #  45/128
            [-5.93749e-03,3.39894e-02,-1.18526e-01,4.32038e-01,7.69305e-01,-1.42566e-01,3.78182e-02,-6.37730e-03], #  46/128
            [-6.01374e-03,3.44568e-02,-1.20382e-01,4.42134e-01,7.61181e-01,-1.43313e-01,3.81111e-02,-6.43358e-03], #  47/128
            [-6.08674e-03,3.49066e-02,-1.22185e-01,4.52218e-01,7.52958e-01,-1.43968e-01,3.83800e-02,-6.48585e-03], #  48/128
            [-6.15644e-03,3.53384e-02,-1.23933e-01,4.62289e-01,7.44637e-01,-1.44531e-01,3.86247e-02,-6.53412e-03], #  49/128
            [-6.22280e-03,3.57519e-02,-1.25624e-01,4.72342e-01,7.36222e-01,-1.45004e-01,3.88454e-02,-6.57836e-03], #  50/128
            [-6.28577e-03,3.61468e-02,-1.27258e-01,4.82377e-01,7.27714e-01,-1.45387e-01,3.90420e-02,-6.61859e-03], #  51/128
            [-6.34530e-03,3.65227e-02,-1.28832e-01,4.92389e-01,7.19116e-01,-1.45682e-01,3.92147e-02,-6.65479e-03], #  52/128
            [-6.40135e-03,3.68795e-02,-1.30347e-01,5.02377e-01,7.10431e-01,-1.45889e-01,3.93636e-02,-6.68698e-03], #  53/128
            [-6.45388e-03,3.72167e-02,-1.31800e-01,5.12337e-01,7.01661e-01,-1.46009e-01,3.94886e-02,-6.71514e-03], #  54/128
            [-6.50285e-03,3.75341e-02,-1.33190e-01,5.22267e-01,6.92808e-01,-1.46043e-01,3.95900e-02,-6.73929e-03], #  55/128
            [-6.54823e-03,3.78315e-02,-1.34515e-01,5.32164e-01,6.83875e-01,-1.45993e-01,3.96678e-02,-6.75943e-03], #  56/128
            [-6.58996e-03,3.81085e-02,-1.35775e-01,5.42025e-01,6.74865e-01,-1.45859e-01,3.97222e-02,-6.77557e-03], #  57/128
            [-6.62802e-03,3.83650e-02,-1.36969e-01,5.51849e-01,6.65779e-01,-1.45641e-01,3.97532e-02,-6.78771e-03], #  58/128
            [-6.66238e-03,3.86006e-02,-1.38094e-01,5.61631e-01,6.56621e-01,-1.45343e-01,3.97610e-02,-6.79588e-03], #  59/128
            [-6.69300e-03,3.88151e-02,-1.39150e-01,5.71370e-01,6.47394e-01,-1.44963e-01,3.97458e-02,-6.80007e-03], #  60/128
            [-6.71985e-03,3.90083e-02,-1.40136e-01,5.81063e-01,6.38099e-01,-1.44503e-01,3.97077e-02,-6.80032e-03], #  61/128
            [-6.74291e-03,3.91800e-02,-1.41050e-01,5.90706e-01,6.28739e-01,-1.43965e-01,3.96469e-02,-6.79662e-03], #  62/128
            [-6.76214e-03,3.93299e-02,-1.41891e-01,6.00298e-01,6.19318e-01,-1.43350e-01,3.95635e-02,-6.78902e-03], #  63/128
            [-6.77751e-03,3.94578e-02,-1.42658e-01,6.09836e-01,6.09836e-01,-1.42658e-01,3.94578e-02,-6.77751e-03], #  64/128
            [-6.78902e-03,3.95635e-02,-1.43350e-01,6.19318e-01,6.00298e-01,-1.41891e-01,3.93299e-02,-6.76214e-03], #  65/128
            [-6.79662e-03,3.96469e-02,-1.43965e-01,6.28739e-01,5.90706e-01,-1.41050e-01,3.91800e-02,-6.74291e-03], #  66/128
            [-6.80032e-03,3.97077e-02,-1.44503e-01,6.38099e-01,5.81063e-01,-1.40136e-01,3.90083e-02,-6.71985e-03], #  67/128
            [-6.80007e-03,3.97458e-02,-1.44963e-01,6.47394e-01,5.71370e-01,-1.39150e-01,3.88151e-02,-6.69300e-03], #  68/128
            [-6.79588e-03,3.97610e-02,-1.45343e-01,6.56621e-01,5.61631e-01,-1.38094e-01,3.86006e-02,-6.66238e-03], #  69/128
            [-6.78771e-03,3.97532e-02,-1.45641e-01,6.65779e-01,5.51849e-01,-1.36969e-01,3.83650e-02,-6.62802e-03], #  70/128
            [-6.77557e-03,3.97222e-02,-1.45859e-01,6.74865e-01,5.42025e-01,-1.35775e-01,3.81085e-02,-6.58996e-03], #  71/128
            [-6.75943e-03,3.96678e-02,-1.45993e-01,6.83875e-01,5.32164e-01,-1.34515e-01,3.78315e-02,-6.54823e-03], #  72/128
            [-6.73929e-03,3.95900e-02,-1.46043e-01,6.92808e-01,5.22267e-01,-1.33190e-01,3.75341e-02,-6.50285e-03], #  73/128
            [-6.71514e-03,3.94886e-02,-1.46009e-01,7.01661e-01,5.12337e-01,-1.31800e-01,3.72167e-02,-6.45388e-03], #  74/128
            [-6.68698e-03,3.93636e-02,-1.45889e-01,7.10431e-01,5.02377e-01,-1.30347e-01,3.68795e-02,-6.40135e-03], #  75/128
            [-6.65479e-03,3.92147e-02,-1.45682e-01,7.19116e-01,4.92389e-01,-1.28832e-01,3.65227e-02,-6.34530e-03], #  76/128
            [-6.61859e-03,3.90420e-02,-1.45387e-01,7.27714e-01,4.82377e-01,-1.27258e-01,3.61468e-02,-6.28577e-03], #  77/128
            [-6.57836e-03,3.88454e-02,-1.45004e-01,7.36222e-01,4.72342e-01,-1.25624e-01,3.57519e-02,-6.22280e-03], #  78/128
            [-6.53412e-03,3.86247e-02,-1.44531e-01,7.44637e-01,4.62289e-01,-1.23933e-01,3.53384e-02,-6.15644e-03], #  79/128
            [-6.48585e-03,3.83800e-02,-1.43968e-01,7.52958e-01,4.52218e-01,-1.22185e-01,3.49066e-02,-6.08674e-03], #  80/128
            [-6.43358e-03,3.81111e-02,-1.43313e-01,7.61181e-01,4.42134e-01,-1.20382e-01,3.44568e-02,-6.01374e-03], #  81/128
            [-6.37730e-03,3.78182e-02,-1.42566e-01,7.69305e-01,4.32038e-01,-1.18526e-01,3.39894e-02,-5.93749e-03], #  82/128
            [-6.31703e-03,3.75010e-02,-1.41727e-01,7.77327e-01,4.21934e-01,-1.16618e-01,3.35046e-02,-5.85804e-03], #  83/128
            [-6.25277e-03,3.71596e-02,-1.40794e-01,7.85244e-01,4.11823e-01,-1.14659e-01,3.30027e-02,-5.77544e-03], #  84/128
            [-6.18454e-03,3.67941e-02,-1.39767e-01,7.93055e-01,4.01710e-01,-1.12650e-01,3.24843e-02,-5.68976e-03], #  85/128
            [-6.11236e-03,3.64044e-02,-1.38644e-01,8.00757e-01,3.91596e-01,-1.10593e-01,3.19495e-02,-5.60103e-03], #  86/128
            [-6.03624e-03,3.59905e-02,-1.37426e-01,8.08348e-01,3.81484e-01,-1.08490e-01,3.13987e-02,-5.50931e-03], #  87/128
            [-5.95620e-03,3.55525e-02,-1.36111e-01,8.15826e-01,3.71376e-01,-1.06342e-01,3.08323e-02,-5.41467e-03], #  88/128
            [-5.87227e-03,3.50903e-02,-1.34699e-01,8.23188e-01,3.61276e-01,-1.04150e-01,3.02507e-02,-5.31716e-03], #  89/128
            [-5.78446e-03,3.46042e-02,-1.33189e-01,8.30432e-01,3.51186e-01,-1.01915e-01,2.96543e-02,-5.21683e-03], #  90/128
            [-5.69280e-03,3.40940e-02,-1.31581e-01,8.37557e-01,3.41109e-01,-9.96402e-02,2.90433e-02,-5.11376e-03], #  91/128
            [-5.59731e-03,3.35600e-02,-1.29874e-01,8.44559e-01,3.31048e-01,-9.73254e-02,2.84182e-02,-5.00800e-03], #  92/128
            [-5.49804e-03,3.30021e-02,-1.28068e-01,8.51437e-01,3.21004e-01,-9.49727e-02,2.77794e-02,-4.89961e-03], #  93/128
            [-5.39500e-03,3.24205e-02,-1.26161e-01,8.58189e-01,3.10980e-01,-9.25834e-02,2.71272e-02,-4.78866e-03], #  94/128
            [-5.28823e-03,3.18153e-02,-1.24154e-01,8.64812e-01,3.00980e-01,-9.01591e-02,2.64621e-02,-4.67520e-03], #  95/128
            [-5.17776e-03,3.11866e-02,-1.22047e-01,8.71305e-01,2.91006e-01,-8.77011e-02,2.57844e-02,-4.55932e-03], #  96/128
            [-5.06363e-03,3.05345e-02,-1.19837e-01,8.77666e-01,2.81060e-01,-8.52109e-02,2.50946e-02,-4.44107e-03], #  97/128
            [-4.94589e-03,2.98593e-02,-1.17526e-01,8.83893e-01,2.71144e-01,-8.26900e-02,2.43930e-02,-4.32052e-03], #  98/128
            [-4.82456e-03,2.91609e-02,-1.15113e-01,8.89984e-01,2.61263e-01,-8.01399e-02,2.36801e-02,-4.19774e-03], #  99/128
            [-4.69970e-03,2.84397e-02,-1.12597e-01,8.95936e-01,2.51417e-01,-7.75620e-02,2.29562e-02,-4.07279e-03], # 100/128
            [-4.57135e-03,2.76957e-02,-1.09978e-01,9.01749e-01,2.41609e-01,-7.49577e-02,2.22218e-02,-3.94576e-03], # 101/128
            [-4.43955e-03,2.69293e-02,-1.07256e-01,9.07420e-01,2.31843e-01,-7.23286e-02,2.14774e-02,-3.81671e-03], # 102/128
            [-4.30435e-03,2.61404e-02,-1.04430e-01,9.12947e-01,2.22120e-01,-6.96762e-02,2.07233e-02,-3.68570e-03], # 103/128
            [-4.16581e-03,2.53295e-02,-1.01501e-01,9.18329e-01,2.12443e-01,-6.70018e-02,1.99599e-02,-3.55283e-03], # 104/128
            [-4.02397e-03,2.44967e-02,-9.84679e-02,9.23564e-01,2.02814e-01,-6.43069e-02,1.91877e-02,-3.41815e-03], # 105/128
            [-3.87888e-03,2.36423e-02,-9.53307e-02,9.28650e-01,1.93236e-01,-6.15931e-02,1.84071e-02,-3.28174e-03], # 106/128
            [-3.73062e-03,2.27664e-02,-9.20893e-02,9.33586e-01,1.83711e-01,-5.88617e-02,1.76185e-02,-3.14367e-03], # 107/128
            [-3.57923e-03,2.18695e-02,-8.87435e-02,9.38371e-01,1.74242e-01,-5.61142e-02,1.68225e-02,-3.00403e-03], # 108/128
            [-3.42477e-03,2.09516e-02,-8.52933e-02,9.43001e-01,1.64831e-01,-5.33522e-02,1.60193e-02,-2.86289e-03], # 109/128
            [-3.26730e-03,2.00132e-02,-8.17385e-02,9.47477e-01,1.55480e-01,-5.05770e-02,1.52095e-02,-2.72032e-03], # 110/128
            [-3.10689e-03,1.90545e-02,-7.80792e-02,9.51795e-01,1.46192e-01,-4.77900e-02,1.43934e-02,-2.57640e-03], # 111/128
            [-2.94361e-03,1.80759e-02,-7.43154e-02,9.55956e-01,1.36968e-01,-4.49929e-02,1.35716e-02,-2.43121e-03], # 112/128
            [-2.77751e-03,1.70776e-02,-7.04471e-02,9.59958e-01,1.27812e-01,-4.21869e-02,1.27445e-02,-2.28483e-03], # 113/128
            [-2.60868e-03,1.60599e-02,-6.64743e-02,9.63798e-01,1.18725e-01,-3.93735e-02,1.19125e-02,-2.13733e-03], # 114/128
            [-2.43718e-03,1.50233e-02,-6.23972e-02,9.67477e-01,1.09710e-01,-3.65541e-02,1.10760e-02,-1.98880e-03], # 115/128
            [-2.26307e-03,1.39681e-02,-5.82159e-02,9.70992e-01,1.00769e-01,-3.37303e-02,1.02356e-02,-1.83931e-03], # 116/128
            [-2.08645e-03,1.28947e-02,-5.39305e-02,9.74342e-01,9.19033e-02,-3.09033e-02,9.39154e-03,-1.68894e-03], # 117/128
            [-1.90738e-03,1.18034e-02,-4.95412e-02,9.77526e-01,8.31162e-02,-2.80746e-02,8.54441e-03,-1.53777e-03], # 118/128
            [-1.72594e-03,1.06946e-02,-4.50483e-02,9.80543e-01,7.44095e-02,-2.52457e-02,7.69462e-03,-1.38589e-03], # 119/128
            [-1.54221e-03,9.56876e-03,-4.04519e-02,9.83392e-01,6.57852e-02,-2.24178e-02,6.84261e-03,-1.23337e-03], # 120/128
            [-1.35627e-03,8.42626e-03,-3.57525e-02,9.86071e-01,5.72454e-02,-1.95925e-02,5.98883e-03,-1.08030e-03], # 121/128
            [-1.16820e-03,7.26755e-03,-3.09503e-02,9.88580e-01,4.87921e-02,-1.67710e-02,5.13372e-03,-9.26747e-04], # 122/128
            [-9.78093e-04,6.09305e-03,-2.60456e-02,9.90917e-01,4.04274e-02,-1.39548e-02,4.27773e-03,-7.72802e-04], # 123/128
            [-7.86031e-04,4.90322e-03,-2.10389e-02,9.93082e-01,3.21531e-02,-1.11453e-02,3.42130e-03,-6.18544e-04], # 124/128
            [-5.92100e-04,3.69852e-03,-1.59305e-02,9.95074e-01,2.39714e-02,-8.34364e-03,2.56486e-03,-4.64053e-04], # 125/128
            [-3.96391e-04,2.47942e-03,-1.07209e-02,9.96891e-01,1.58840e-02,-5.55134e-03,1.70888e-03,-3.09412e-04], # 126/128
            [-1.98993e-04,1.24642e-03,-5.41054e-03,9.98534e-01,7.89295e-03,-2.76968e-03,8.53777e-04,-1.54700e-04], # 127/128
            [ 0.00000e+00,0.00000e+00,0.00000e+00,1.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00], # 128/128
        ]))

    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample

        x_orig_shape = x.shape

        if self.taps.device != x.device: 
            self.taps = self.taps.to(x.device)

        initial_mu = torch.rand(x_orig_shape[:-1], device=x.device) if self.initial_offset else torch.zeros(x_orig_shape[:-1], device=x.device)
        initial_sro = torch.normal(0, self.std_dev_init_hz, x_orig_shape[:-1], device=x.device)

        sro = torch.normal(0, self.std_dev_walk_hz, x_orig_shape, device=x.device)
        sro = torch.cumsum(sro, 1) + initial_sro[..., None]

        mu_inc = 1.0 + sro / self.samp_rate
        s = initial_mu[..., None] + torch.cumsum(mu_inc, -1)
        f = torch.floor(s)
        mu = s - f

        ii = (f).to(int)
        imu = torch.round(mu * (len(self.taps)-1)).to(int)

        idx = ii+self.taps.shape[1] < x_orig_shape[-1]-1 # Find all indices of x that will not overrun
        idx = idx.flatten(0,-2)[torch.argmin(idx.sum(-1))] # select the safest channel indexing
        imu = imu[...,idx]
        ii = ii[...,idx]

        append_dir = torch.randint(2, (1,), dtype=torch.bool, generator=self.generator, device=x.device) # Randomly choose to prepend or postpend zeros 
        pad = x_orig_shape[-1]-ii.shape[-1]  # calculate size lost from sample rate changes

        a = x.gather(-1, ii) # filter x
        a = torch.nn.functional.pad(a, (4,pad+3) if append_dir else (pad+3,4)) # Pad a to retain input shape
        a = a.unfold(-1, 8, 1) # Unflold a for convolution

        imu = torch.nn.functional.pad(imu, (0,pad) if append_dir else (pad,0)) # Pad imu to retain input shape on filter
        b = self.taps[imu] # filter taps to match a shape

        x = torch.sum(a * b, dim=-1) # Perform convolution

        return (x, y, t0)

class CFOTransform:
    def __init__(self, sample_rate_hz: float, walk_std_hz: float, init_std_hz: float, generator: Optional[torch.Generator]=None):
        self.samp_rate = sample_rate_hz
        self.std_dev_walk_hz = walk_std_hz
        self.std_dev_init_hz = init_std_hz
        self.generator = generator

    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample

        initial_cfo = torch.normal(0, self.std_dev_init_hz, (*x.shape[:-1], 1), generator=self.generator, device=x.device)
        initial_angle = 0
        
        # update cfo
        cfo = torch.normal(0, self.std_dev_walk_hz, x.shape, generator=self.generator, device=x.device)
        cfo = torch.cumsum(cfo, -1) + initial_cfo

        # update and wrap angle
        angle = 2 * torch.pi * cfo / self.samp_rate
        angle = torch.cumsum(angle, -1) + initial_angle
        angle = torch.remainder(angle, 2*torch.pi) # limit to 0:2pi range
        x = x * (torch.cos(angle)+torch.sin(angle)*1j)

        return (x, y, t0)

class SelectiveFaderTransform:
    """
    A. Alimohammad et al, "Compact Rayleigh and Rician fading simulator based on random walk processes," 
    IET Communications, vol. 3, (8), pp. 1333-1342, 2009. 
    Available: https://link.ezproxy.neu.edu/login?url=https://www.proquest.com/scholarly-journals/compact-rayleigh-rician-fading-simulator-based-on/docview/1617053209/se-2.
    """
    def __init__(self, N: int, fDTs: float, LOS: bool, K: float, delays: List[float], mags: List[float], ntaps: int, n_rx: int, generator: Optional[torch.Generator]=None):
        if (len(mags) != len(delays)):
            raise RuntimeError("magnitude and delay vectors must be the same length!")
        if (ntaps < 1):
            raise RuntimeError("ntaps must be >= 1")

        self.d_k = K
        self.d_fDTs = fDTs
        self.n_taps = ntaps
        self.n_rx = n_rx
        self.n_faders = len(delays)
        self.generator = generator

        d_delays = torch.tensor(delays)[...,None]
        d_mags = torch.tensor(mags)[...,None]
        self.interpmag = torch.arange(self.n_taps, dtype=torch.float32).flipud().repeat(self.n_faders,1)
        self.interpmag = torch.sinc(torch.pi * (self.interpmag - d_delays))
        self.interpmag = self.interpmag * d_mags

        self.d_N = N
        self.d_step = (0.00125 * self.d_fDTs)**1.1 # max step size approximated from Table 2

        self.scale_sin = np.sqrt(2.0 / self.d_N)
        self.scale_los = np.sqrt(self.d_k) / np.sqrt(self.d_k + 1)
        self.scale_nlos = 1 / np.sqrt(self.d_k + 1)
        self.d_LOS = LOS
        
    def __call__(self, sample: Tuple[Tensor]) -> Tuple[Tensor]:
        x, y, t0 = sample
        n_samples = x.shape[-1]
        new_shape = x.shape[:-1]

        if self.interpmag.device != x.device: 
            self.interpmag = self.interpmag.to(x.device)

        # update_theta()
        sign = torch.randint(2, (*new_shape,1,self.n_faders,1), dtype=torch.bool, generator=self.generator, device=x.device)
        step_dir = torch.ones((*new_shape,1,self.n_faders,1), device=x.device)
        step_dir[sign] = -1
        theta = torch.rand((*new_shape, n_samples, self.n_faders, 1), generator=self.generator, device=x.device) * step_dir * self.d_step
        initial_theta = (2*torch.pi) * torch.rand((*new_shape, 1, self.n_faders, 1), generator=self.generator, device=x.device) # 0:2pi
        theta = torch.cumsum(theta, -2) + initial_theta
        theta = torch.remainder(theta, torch.tensor([2*torch.pi], device=x.device)) - torch.pi

        init_phase = (-2*torch.pi) * torch.rand((*new_shape,1,self.n_faders,self.d_N,2), generator=self.generator, device=x.device) + torch.pi

        alpha_n = torch.arange(1, self.d_N+1, device=x.device).repeat((*new_shape,n_samples,self.n_faders,1))
        alpha_n = (2 * torch.pi * alpha_n - torch.pi + theta) / (4 * self.d_N)
        doppler_shift = torch.stack([torch.cos(alpha_n), torch.sin(alpha_n)], -1)
        m = torch.arange(1,n_samples+1, device=x.device)[...,None]
        doppler_shift = 2 * torch.pi * self.d_fDTs * m[...,None,None] * doppler_shift
        doppler_shift_cumsum = torch.cumsum(doppler_shift, -3)

        phase = (doppler_shift_cumsum + init_phase) % (2 * torch.pi)
        phase = torch.cos(phase)
        fading_taps = torch.sum(torch.view_as_complex(phase), -1, keepdim=True) * self.scale_sin

        if (self.d_LOS):
            theta_los = (-2*torch.pi) * torch.rand((*new_shape, 1, self.n_faders), generator=self.generator, device=x.device) + torch.pi
            psi = (-2*torch.pi) * torch.rand((*new_shape,1,self.n_faders), generator=self.generator, device=x.device) + torch.pi
            psi_los_inc = 2 * torch.pi * self.d_fDTs * m * torch.cos(theta_los)
            psi = (psi + psi_los_inc) % (2 * torch.pi)
            los = torch.cos(psi) * (torch.sin(psi)*1j)
            fading_taps = fading_taps * self.scale_nlos + self.scale_los * los[...,None]

        # sum each flat fading component as the taps
        taps = torch.sum(fading_taps * self.interpmag, -2)

        # # Convolution
        a = torch.nn.functional.pad(x, (self.n_taps//2,self.n_taps//2)) # Pad x to retain input shape
        a = a.unfold(-1, self.n_taps, 1)
        x = torch.sum(a * taps, dim=-1)

        return (x, y, t0)

class DisAMRDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size, frame_size: int = 1024, n_rx: int = 1, min_snr = 0, max_snr = 13, seed: int = 42):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.n_rx = n_rx
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.transforms = []
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

        # gens = [torch.Generator().manual_seed(self.rng.seed()+i) for i in range(self.n_rx)]
        gens = torch.Generator(device='cuda').manual_seed(self.rng.seed()+1)

        fs = 200e3
        sro_walk_std = 0.01
        sro_init_std = 1e2
        cfo_walk_std_hz = 0.01
        cfo_init_std_hz = 1e3
        n_sinusoids = 8 # Number of sinusoids used to simulate gain on each ray
        fDTs = 1.0/fs # Normalized Max Doppler (fD/fs)
        LOS_present = True # LOS path exists? chooses Rician (LOS) vs Rayleigh (NLOS) model.
        rician_factor_K = 4 # Rician factor (ratio of the specular power to the scattered power)
        pdp_delays = [0.0, 0.9, 1.7] # Time delay in the fir filter (in samples) for each arriving WSSUS Ray
        pdp_mags = [1, 0.8, 0.3] # Magnitude corresponding to each WSSUS Ray (linear)
        n_taps = 7 # Number of FIR taps to use in selective fading model

        self.transforms = [
            MultiRxTransform(self.n_rx),
            SROTransform(fs, sro_walk_std, sro_init_std, generator=gens),
            CFOTransform(fs, cfo_walk_std_hz, cfo_init_std_hz, generator=gens),
            SelectiveFaderTransform(n_sinusoids, fDTs, LOS_present, rician_factor_K, pdp_delays, pdp_mags, n_taps, self.n_rx, gens),
            MultichannelNoiseTransform(self.min_snr, self.max_snr, gens),
        ]

        ds_full = TensorDataset(x, y, t0)

        self.ds_train, self.ds_val, self.ds_test = random_split(ds_full, [0.6, 0.2, 0.2], generator = self.rng)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if type(batch) is list:
            for t in self.transforms:
                batch = t(batch)
        return batch

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
