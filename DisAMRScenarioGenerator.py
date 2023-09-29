import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.patches as mpatches
from enum import IntEnum
from scipy.spatial.distance import pdist, cdist
from numpy.lib.stride_tricks import sliding_window_view
import pyfastnoisesimd
from matplotlib.lines import Line2D
import random
import json

DEFAULT_MAP_CONFIG = json.load(open("default_map_config.json"))


class TerrainType(IntEnum):
    Open = 1
    Foliage = 2
    Suburban = 3
    Urban = 4


class DisAMRScenarioGenerator:
    def __init__(self, n_receivers=1, resolution=1024, min_dist=5, seed=42) -> None:
        self.map = []
        self.transmitters = []
        self.receivers = []
        self.n_rx = n_receivers
        self.n_tx = 1
        self.resolution = resolution
        self.min_dist = min_dist
        self.rng = np.random.default_rng(seed=seed)
        self.fns = pyfastnoisesimd.Noise(seed=seed, numWorkers=1)
        self.fns.noiseType = pyfastnoisesimd.NoiseType.Simplex

        self.RegenerateFullScenario()

    def RegenerateFullScenario(
        self,
        foliage_freq=DEFAULT_MAP_CONFIG["foliage_freq"],
        foliage_octaves=DEFAULT_MAP_CONFIG["foliage_octaves"],
        foliage_octaves_factor=DEFAULT_MAP_CONFIG["foliage_octaves_factor"],
        foliage_levels=DEFAULT_MAP_CONFIG["foliage_levels"],
        foliage_redistribution=DEFAULT_MAP_CONFIG["foliage_redistribution"],
        pop_octaves=DEFAULT_MAP_CONFIG["pop_octaves"],
        pop_freq=DEFAULT_MAP_CONFIG["pop_freq"],
        pop_octaves_factor=DEFAULT_MAP_CONFIG["pop_octaves_factor"],
        pop_levels=DEFAULT_MAP_CONFIG["pop_levels"],
        pop_redistribution=DEFAULT_MAP_CONFIG["pop_redistribution"],
        sender_in_city=True,
    ):
        pop_map = self._createMap(
            pop_freq,
            self.resolution,
            pop_octaves,
            pop_octaves_factor,
            pop_redistribution,
            pop_levels,
            island=True,
        )
        foliage_map = self._createMap(
            foliage_freq,
            self.resolution,
            foliage_octaves,
            foliage_octaves_factor,
            foliage_redistribution,
            foliage_levels,
            island=False,
        )
        self.map = self._resolveMap(pop_map, foliage_map)

        self.transmitters, self.receivers = self.create_nodes(
            sender_in_city, self.map, self.resolution
        )
        self.receivers = self.RegenerateReceivers(self.transmitters, self.receivers)

    def create_nodes(self, sender_in_city, map, map_resolution):
        if sender_in_city:
            city_points = np.stack(np.where(map == TerrainType.Urban)).T
            transmitters = city_points[
                np.random.randint(len(city_points), size=(self.n_tx)), :
            ]
            transmitters[:, [0, 1]] = transmitters[
                :, [1, 0]
            ]  # Swap colums for consistency during unpack
        else:
            transmitters = np.random.randint(map_resolution, size=(self.n_tx, 2))
        receivers = np.random.randint(map_resolution, size=(self.n_rx, 2))

        return transmitters, receivers

    def RegenerateReceivers(self, transmitters, receivers):
        min_dist = self.min_dist - 1
        while min_dist < self.min_dist:
            receivers = np.random.randint(self.resolution, size=(self.n_rx, 2))
            min_dist = cdist(transmitters, receivers, "euclidean").min()
        return receivers

    def GetScenario(
        self,
    ):
        return self._computeDistances(self.transmitters, self.receivers, self.map)

    def PlotMap(self, save_path=None):
        fig, axes = plt.subplots(nrows=1, figsize=(10, 10))
        im = axes.imshow(self.map, origin="lower", cmap="Blues")
        values = np.unique(self.map.ravel())

        for d in self._computeDistances(self.transmitters, self.receivers, self.map):
            key_points = d["key_points"]
            xtrans_coords, ytrans_coords = key_points[:, 0], key_points[:, 1]
            xSender, ySender = d["sender_coords"]
            xReciever, yReciever = d["reciever_coords"]
            axes.plot(xtrans_coords, ytrans_coords, "rx-", zorder=0)
            axes.scatter(xSender, ySender, marker="o", color="y", s=50, zorder=10)
            axes.scatter(xReciever, yReciever, marker="o", color="g", s=50, zorder=10)

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
                    label="Reciever",
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

    def _createMap(
        self,
        freq,
        map_resolution,
        octaves,
        octave_factor,
        redistribution,
        levels,
        island,
    ):
        # Initialize map
        z = np.zeros((map_resolution, map_resolution))

        # Apply noise maps with different frequencies
        of = 1
        for _ in range(octaves):
            self.fns.seed = self.fns.seed + 1
            self.fns.frequency = freq * of
            z += (1 / of) * self.fns.genAsGrid([map_resolution, map_resolution])
            of *= octave_factor

        # Enable population islands
        znorm = (2 * (z - np.min(z)) / (np.max(z) - np.min(z))) - 1  # norm from -1 to 1
        _b = np.linspace(-1, 1, map_resolution)
        _x, _y = np.meshgrid(_b, _b)
        d = -(1 - ((1 - _x**2) * (1 - _y**2)))
        if island:
            d *= -1
        z = (znorm + (1 - d)) / 2

        # Control diff between peaks and troughs
        z = (z - np.min(z)) / (np.max(z) - np.min(z))  # norm from 0 to 1
        z = np.power(z, redistribution)

        # Quantize to specific terrain levels
        z = np.digitize(z, np.linspace(z.min(), z.max() + 0.0000001, levels + 1))
        z = (z - np.min(z)) / (np.max(z) - np.min(z))  # norm from 0 to 1

        return z

    def _resolveMap(self, pop_map, foliage_map):
        assert pop_map.shape == foliage_map.shape
        map = np.ones(pop_map.shape)
        # map[pop_map == 0] = float(TerrainType.Open)  # Terrain
        map[np.where(foliage_map > 0.7)] = float(TerrainType.Foliage)  # Foliage
        # map[np.where((pop_map >= 0.8) & (foliage_map <= 0.5))] = float(TerrainType.Urban)  # Urban
        map[np.where(pop_map >= 0.7)] = float(TerrainType.Suburban)
        map[np.where(pop_map >= 0.9)] = float(TerrainType.Urban)
        return map

    def _computeDistances(
        self,
        senders,
        recievers,
        map,
        distance_metric="euclidean",
        path_aggregation=True,
        distance_aggregation_threshold=0.0187,
    ):
        path_data = []
        for sender, reciever in product(senders, recievers):
            xSender, ySender = sender
            xReciever, yReciever = reciever
            num = 10000
            # Evaluate points between sender and reciever
            x, y = np.linspace(xSender, xReciever, num), np.linspace(
                ySender, yReciever, num
            )
            # Extract the values along the line
            zi = map[y.astype(int), x.astype(int)]
            terrain_transitions = (np.abs(zi[1:] - zi[:-1]) > 0).astype(int)
            terrain_transitions = np.concatenate([np.array([0]), terrain_transitions])
            xtrans_coords, ytrans_coords = (
                x[terrain_transitions == 1],
                y[terrain_transitions == 1],
            )
            xtrans_coords = np.concatenate([[xSender], xtrans_coords, [xReciever]])
            ytrans_coords = np.concatenate([[ySender], ytrans_coords, [yReciever]])
            # terrain type between last transition and reciever is the same
            terrain_types = map[ytrans_coords.astype(int), xtrans_coords.astype(int)][
                :-1
            ]
            key_points = np.stack([xtrans_coords, ytrans_coords], axis=1)
            point_pairs = sliding_window_view(key_points, window_shape=(2, 2)).squeeze()
            if point_pairs.shape == (2, 2):
                point_pairs = point_pairs[np.newaxis, ...]
            distances = np.array(
                [pdist(pair, metric=distance_metric) for pair in point_pairs]
            )
            new_path = {
                "key_points": key_points,
                "reciever_coords": reciever,
                "sender_coords": sender,
                "terrain_type": terrain_types,
                "distances": distances,
            }
            if not path_aggregation:
                path_data.append(new_path)
                continue

            map_diagonal = np.sqrt(np.sum(np.power(map.shape, 2)))
            min_distance = distance_aggregation_threshold * map_diagonal
            (min_distance_idxs,) = np.where(distances.flatten() < min_distance)
            new_point_pairs = [
                pair
                for pair_idx, pair in enumerate(point_pairs)
                if pair_idx not in min_distance_idxs
            ]
            new_point_pairs = np.array(new_point_pairs)
            for pair_idx in range(len(new_point_pairs) - 1):
                new_point_pairs[pair_idx, -1] = new_point_pairs[pair_idx + 1, 0]
            key_points, ind = np.unique(
                np.concatenate(
                    [
                        sender.reshape(-1, 2),
                        new_point_pairs.reshape(-1, 2),
                        reciever.reshape(-1, 2),
                    ],
                    axis=0,
                ),
                axis=0,
                return_index=True
            )
            key_points = key_points[np.argsort(ind)]
            point_pairs = sliding_window_view(key_points, window_shape=(2, 2)).squeeze()
            distances = np.array(
                [pdist(pair, metric=distance_metric) for pair in point_pairs]
            )
            # Terrain types for small distance sections defaults to last distance's terrain type
            terrain_types = [
                t for idx, t in enumerate(terrain_types) if idx not in min_distance_idxs
            ]
            new_path = {
                "key_points": key_points,
                "reciever_coords": reciever,
                "sender_coords": sender,
                "terrain_type": terrain_types,
                "distances": distances,
            }
            path_data.append(new_path)
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in new_path.items()}, open("path_data_w_agg.json", "w"), indent=4)
        return path_data


if __name__ == "__main__":
    seed = 163250513
    np.random.seed(seed)
    random.seed(seed)
    print("Generating map")
    scen_gen = DisAMRScenarioGenerator(seed=seed)
    scen_gen.PlotMap(save_path="C_.png")
