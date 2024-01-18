import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .MapGenerator import MapGenerator, TerrainType
from .PathOptimizer import OptimizeReceivers

class ScenarioGenerator:
    def __init__(
        self,
        n_receivers=1,
        n_transmitters=1,
        batch_size=1,
        h_receivers = 10,
        h_transmitters = 1.5,
        resolution=500,
        min_receiver_dist=200,
        min_path_distance=50,
        seed=42,
        n_workers=1
    ) -> None:
        self.map = []
        self.transmitters = []
        self.receivers = []
        self.n_rx = n_receivers
        self.n_tx = n_transmitters
        self.batch_size = batch_size
        self.h_tx = h_transmitters
        self.h_rx = h_receivers
        self.resolution = resolution
        self.min_receiver_dist = min_receiver_dist
        self.min_path_distance = min_path_distance
        self.seed=seed
        self.rng = np.random.default_rng(seed=seed)
        self.map_gen = MapGenerator(resolution, n_workers=n_workers, seed=seed)

        self.RegenerateFullScenario()

    def RegenerateFullScenario(self, target_path_loss=None):
        map_diagonal = np.sqrt(np.sum(np.power(self.resolution, 2)))
        if self.min_receiver_dist > map_diagonal:
            raise Exception("Invalid min receiver distance and map size specified")

        if self.min_path_distance > map_diagonal:
            raise Exception("Invalid min path distance and map size specified")
        
        self.map = self.map_gen() #Sample map generator

        self._create_nodes()

        if target_path_loss is not None:
            diff = OptimizeReceivers(target_path_loss)
            return diff
        else:
            return
    
    def _create_nodes(self):
        self.transmitters = self.rng.integers(self.resolution, size=(self.batch_size, self.n_tx, 3)).astype(np.float32)
        self.transmitters[...,-1] = self.h_tx

        # Regenerate receivers until all meet the minimum distance requirement
        self.receivers = self.rng.integers(self.resolution, size=(self.batch_size, self.n_rx, 3)).astype(np.float32)
        while torch.cdist(torch.from_numpy(self.transmitters[...,:2]).float(), torch.from_numpy(self.receivers[...,:2]).float(), 2).numpy().min() < self.min_receiver_dist: # torc.cdist
            self.receivers = self.rng.integers(0, self.resolution, size=(self.batch_size, self.n_rx, 3)).astype(np.float32)
        self.receivers[...,-1] = self.h_rx

    def PlotMap(self, save_path=None):
        fig, axes = plt.subplots(nrows=1, figsize=(10, 10))
        im = axes.imshow(self.map, origin="lower", cmap="Blues")
        values = np.unique(self.map.ravel())

        for j in range(self.batch_size):
            for i in range(self.n_rx):
                axes.scatter(self.receivers[j, i,0], self.receivers[j, i,1], marker="o", color="g", s=50, zorder=10)

            for i in range(self.n_tx):
                axes.scatter(self.transmitters[j, i, 0], self.transmitters[j, i, 1], marker="o", color="y", s=50, zorder=10)

        # key_points = d["key_points"]
        # xtrans_coords, ytrans_coords = key_points[:, 0], key_points[:, 1]
        # axes.plot(xtrans_coords[::-1], ytrans_coords[::-1], "rx-", zorder=0)

        axes.axis("image")

        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [
            mpatches.Patch(
                color=colors[v_idx],
                label=f"{str(TerrainType(v)).split('.')[-1]} Terrain",
            )
            for v_idx, v in enumerate(values)
        ]
        patches.extend(
            [
                Line2D(
                    [],
                    [],
                    label="receiver",
                    color="white",
                    marker="o",
                    markersize=12,
                    markerfacecolor="g",
                )
            ]
        )
        patches.extend(
            [
                Line2D(
                    [],
                    [],
                    label="Sender",
                    color="white",
                    marker="o",
                    markersize=12,
                    markerfacecolor="y",
                )
            ]
        )
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.grid(True)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            return fig

        plt.show()
        return fig


import numpy as np
if __name__ == "__main__":
    seed = 46

    scenarioGen = ScenarioGenerator(n_receivers=6, resolution=500, min_receiver_dist=1000, seed=seed)
    scenarioGen.PlotMap(save_path="Map.png")