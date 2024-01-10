from enum import IntEnum
import numpy as np
import pyfastnoisesimd

DEFAULT_MAP_CONFIG = {
    "foliage_freq": 0.005,
    "foliage_octaves": 3,
    "foliage_octaves_factor": 7.1,
    "foliage_levels": 10,
    "foliage_redistribution": 0.7,
    "pop_octaves": 10,
    "pop_freq": 0.001,
    "pop_octaves_factor": 2.61,
    "pop_levels": 11,
    "pop_redistribution": 4
}

class TerrainType(IntEnum):
    Urban = 1
    Rural = 0

class MapGenerator():
    def __init__(self,
                  width: int,
                  freq=DEFAULT_MAP_CONFIG["pop_freq"],
                  octaves=DEFAULT_MAP_CONFIG["pop_octaves"],
                  octaves_factor=DEFAULT_MAP_CONFIG["pop_octaves_factor"],
                  levels=DEFAULT_MAP_CONFIG["pop_levels"],
                  redistribution=DEFAULT_MAP_CONFIG["pop_redistribution"],
                  island=True,
                  urban_threshold=0.5,
                  n_workers=1,
                  seed=42) -> None:
        self.width = width
        self.freq = freq
        self.octaves = octaves
        self.octaves_factor = octaves_factor
        self.levels = levels
        self.redistribution = redistribution
        self.island = island
        self.urban_threshold = urban_threshold

        self.fns = pyfastnoisesimd.Noise(seed=seed, numWorkers=n_workers)
        self.fns.noiseType = pyfastnoisesimd.NoiseType.Simplex

    def __call__(self):
        # Initialize map
        z = np.zeros((self.width, self.width))

        # Apply noise maps with different frequencies
        of = 1
        for _ in range(self.octaves):
            self.fns.seed = self.fns.seed + 1
            self.fns.frequency = self.freq * of
            z += (1 / of) * self.fns.genAsGrid([self.width, self.width])
            of *= self.octaves_factor

        # Enable population islands
        znorm = (2 * (z - np.min(z)) / (np.max(z) - np.min(z))) - 1  # norm from -1 to 1
        _b = np.linspace(-1, 1, self.width)
        _x, _y = np.meshgrid(_b, _b)
        d = -(1 - ((1 - _x**2) * (1 - _y**2)))
        if self.island:
            d *= -1
        z = (znorm + (1 - d)) / 2

        # Control diff between peaks and troughs
        z = (z - np.min(z)) / (np.max(z) - np.min(z))  # norm from 0 to 1
        z = np.power(z, self.redistribution)

        # Quantize to specific terrain levels
        z = np.digitize(z, np.linspace(z.min(), z.max() + 0.0000001, self.levels + 1))
        z = (z - np.min(z)) / (np.max(z) - np.min(z))  # norm from 0 to 1

        # Resolve map to classes
        return np.where(z >= self.urban_threshold, TerrainType.Urban, TerrainType.Rural)
