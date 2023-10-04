import numpy as np
import matplotlib

matplotlib.use("Agg")
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
import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
from math import inf, sqrt
import pandas as pd
from copy import deepcopy

DEFAULT_MAP_CONFIG = json.load(open("default_map_config.json"))
MAGIC_CONSTANT = 0


class TerrainType(IntEnum):
    Urban = 2
    Rural = 1


class Direction(IntEnum):
    Up = 1
    Down = 2
    Left = 3
    Right = 4
    Towards = 5
    Away = 6
    Clockwise = 7
    CounterClockwise = 8


class DisAMRScenarioGenerator:
    def __init__(
        self,
        n_receivers=1,
        resolution=500,
        min_receiver_dist=2000,
        min_path_distance=100,
        seed=42,
        meters_per_pixel=10,
        max_received_power=-5550,
    ) -> None:
        self.map = []
        self.transmitters = []
        self.receivers = []
        self.n_rx = n_receivers
        self.n_tx = 1
        self.resolution = resolution
        self.min_receiver_dist = min_receiver_dist / meters_per_pixel
        self.min_path_distance = min_path_distance / meters_per_pixel
        self.rng = np.random.default_rng(seed=seed)
        self.fns = pyfastnoisesimd.Noise(seed=seed, numWorkers=1)
        self.fns.noiseType = pyfastnoisesimd.NoiseType.Simplex
        self.mpp = meters_per_pixel

        self.RegenerateFullScenario(max_received_power)

    def RegenerateFullScenario(
        self,
        max_received_power,
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
        senders_in_city=True,
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

        map_diagonal = np.sqrt(np.sum(np.power(self.map.shape, 2)))

        if self.min_receiver_dist > map_diagonal:
            raise Exception("Invalid min receiver distance and map size specified")

        if self.min_path_distance > map_diagonal:
            raise Exception("Invalid min path distance and map size specified")

        self.transmitters, self.receivers = self.create_nodes(
            senders_in_city, self.map, self.resolution
        )
        self.receivers = self.RegenerateReceivers(
            self.transmitters, self.receivers, max_received_power
        )

    def create_nodes(self, senders_in_city, map, map_resolution):
        if senders_in_city:
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

    def move(self, from_coord, direction, relative_to_coord=None, magnitude=1):
        x, y = from_coord
        if relative_to_coord is None:
            if direction in [Direction.Left, Direction.Right]:
                magnitude = -magnitude if direction == Direction.Left else magnitude
                x_delta = magnitude
                y_delta = 0
            if direction in [Direction.Up, Direction.Down]:
                magnitude = -magnitude if direction == Direction.Down else magnitude
                x_delta = 0
                y_delta = magnitude
        else:
            if direction in [Direction.Towards, Direction.Away]:
                x_target, y_target = relative_to_coord

                eps = np.finfo(np.float32).eps
                hyp_c = sqrt((y_target - y) ** 2 + (x_target - x) ** 2) + eps
                x_delta = magnitude * (x_target - x + eps) / hyp_c
                y_delta = magnitude * (y_target - y + eps) / hyp_c
                if direction == Direction.Away:
                    x_delta *= -1
                    y_delta *= -1

            elif direction in [Direction.Clockwise, Direction.CounterClockwise]:
                x_target, y_target = relative_to_coord
                x_target, y_target = x - x_target, y - y_target
                if direction == Direction.CounterClockwise:
                    x_target, y_target = -y_target, x_target
                else:
                    x_target, y_target = y_target, -x_target
                target = np.array([x_target, y_target])
                target = target / np.linalg.norm(target)
                x_delta = target[0] * magnitude
                y_delta = target[-1] * magnitude

        x += x_delta
        y += y_delta

        new_coord = np.array([x, y])
        new_coord = np.clip(new_coord, 0, self.resolution - 1)

        return new_coord

    def RegenerateReceivers(
        self,
        transmitters,
        receivers,
        max_received_power,
        error=0.01,
        initial_magnitude=1,
        iteration_max=100,
        save_metrics=True,
    ):
        dist = cdist(transmitters, receivers, "euclidean").min()
        while dist < self.min_receiver_dist:
            receivers = np.random.randint(0, self.resolution, size=(self.n_rx, 2))
            dist = cdist(transmitters, receivers, "euclidean").min()
        if max_received_power == inf:
            return receivers

        path_data = self._computeDistances(transmitters, receivers, self.map)
        power = self.calc_scenario_received_power(path_data)
        iteration_idx = 0
        reciever_lengths = []
        while abs(power - max_received_power) > error and iteration_idx < iteration_max:
            magnitude = initial_magnitude * np.exp(-iteration_idx / iteration_max)
            d = {}
            d["magnitude"] = magnitude
            d["error"] = abs(power - max_received_power)
            for i in range(len(path_data)):
                sender, receiver = (
                    path_data[i]["sender_coords"],
                    path_data[i]["receiver_coords"],
                )
                power = self.calc_scenario_received_power(path_data)
                diff = power - max_received_power
                direction = Direction.Away if diff > 0 else Direction.Towards
                receiver = self.move(
                    receiver, direction, magnitude=magnitude, relative_to_coord=sender
                )
                dist = cdist([sender], [receiver], "euclidean").item()
                if dist < self.min_receiver_dist:
                    continue
                receiver_data = next(
                    iter(
                        self._computeDistances(
                            [sender], [receiver], self.map, path_aggregation=True
                        )
                    )
                )
                d[f"receiver_{i}"] = sum(path_data[i]["distances"]).item()
                d[f"direction_{i}"] = 1 if direction == Direction.Away else -1
                path_data[i] = receiver_data

            power = self.calc_scenario_received_power(path_data)
            d["power"] = power
            reciever_lengths.append(d)
            if save_metrics:
                pd.DataFrame.from_records(reciever_lengths).to_csv("test.csv")
            receivers = [p["receiver_coords"] for p in path_data]
            iteration_idx += 1
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
            xReceiver, yReceiver = d["receiver_coords"]
            axes.plot(xtrans_coords[::-1], ytrans_coords[::-1], "rx-", zorder=0)
            axes.scatter(xSender, ySender, marker="o", color="y", s=50, zorder=10)
            axes.scatter(xReceiver, yReceiver, marker="o", color="g", s=50, zorder=10)

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
        map[np.where(pop_map >= 0.5)] = float(TerrainType.Urban)
        map[np.where(pop_map < 0.5)] = float(TerrainType.Rural)
        return map

    def _computeDistances(
        self,
        senders,
        receivers,
        map,
        distance_metric="euclidean",
        path_aggregation=True,
    ):
        path_data = []
        for sender, receiver in product(senders, receivers):
            xSender, ySender = sender
            xReceiver, yReceiver = receiver
            num = 10000
            # Evaluate points between sender and receiver
            x, y = np.linspace(xSender, xReceiver, num), np.linspace(
                ySender, yReceiver, num
            )
            # Extract the values along the line
            zi = map[y.astype(int), x.astype(int)]
            terrain_transitions = (np.abs(zi[1:] - zi[:-1]) > 0).astype(int)
            terrain_transitions = np.concatenate([np.array([0]), terrain_transitions])
            xtrans_coords, ytrans_coords = (
                x[terrain_transitions == 1],
                y[terrain_transitions == 1],
            )
            xtrans_coords = np.concatenate([[xSender], xtrans_coords, [xReceiver]])
            ytrans_coords = np.concatenate([[ySender], ytrans_coords, [yReceiver]])
            # terrain type between last transition and receiver is the same
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
                "receiver_coords": receiver,
                "sender_coords": sender,
                "terrain_type": terrain_types,
                "distances": distances,
            }
            old_distances = distances.sum()
            if not path_aggregation:
                path_data.append(new_path)
                continue

            # Filter out pairs with small distances
            # TODO: Is this arbitrary?
            # maximum min segment possible without eliminating all segments
            min_distance = self.min_path_distance / len(distances)
            (min_distance_idxs,) = np.where(distances.flatten() < min_distance)
            new_point_pairs = [
                pair
                for pair_idx, pair in enumerate(point_pairs)
                if pair_idx not in min_distance_idxs
            ]
            # Assume terrain types for small distance sections defaults to nearest largest distance's terrain type
            terrain_types = [
                t for idx, t in enumerate(terrain_types) if idx not in min_distance_idxs
            ]
            # Force sender and receiver at first and last pairs
            new_point_pairs[0] = np.stack((sender, new_point_pairs[0][-1]))
            new_point_pairs[-1] = np.stack((new_point_pairs[-1][0], receiver))
            new_point_pairs = np.array(new_point_pairs)

            # Connect pairs
            for pair_idx in range(len(new_point_pairs) - 1):
                new_point_pairs[pair_idx, -1] = new_point_pairs[pair_idx + 1, 0]

            # Further consolidate pairs based on terrain equivalence
            temp = []
            i = 0
            while i < len(new_point_pairs):
                j = i + 1
                while j < len(new_point_pairs) and terrain_types[i] == terrain_types[j]:
                    j += 1
                temp.append(
                    np.stack((new_point_pairs[i][0], new_point_pairs[j - 1][-1]))
                )
                i = j
            new_point_pairs = temp

            # Extract key points
            key_points, ind = np.unique(
                np.concatenate(new_point_pairs), axis=0, return_index=True
            )
            key_points = key_points[np.argsort(ind)]

            # Recompute distances
            point_pairs = sliding_window_view(key_points, window_shape=(2, 2)).squeeze(
                axis=1
            )
            distances = np.array(
                [pdist(pair, metric=distance_metric) for pair in point_pairs]
            )

            new_path = {
                "key_points": key_points,
                "receiver_coords": receiver,
                "sender_coords": sender,
                "terrain_type": terrain_types,
                "distances": distances,
            }
            new_distances = distances.sum()
            if abs(old_distances-new_distances) > 0.01:
                raise Exception
            path_data.append(new_path)
        return path_data

    def rural_path_loss(self, distance_2d, distance_3d, fc_ghz, h_ut, los):
        h_bs = 25
        average_building_height = 5.0
        average_street_width = 20

        # LOS condition
        # los_probability = np.exp(-(distance_2d-10.0)/1000.0)
        # los_probability = np.where(distance_2d < 10.0, 1.0, los_probability)
        # los = np.random.binomial(1, los_probability) # Bernoulli Distribution

        # Beak point distance
        # For this computation, the carrifer frequency needs to be in Hz
        distance_breakpoint = (
            2.0 * scipy.constants.pi * h_bs * h_ut * fc_ghz * 1e9 / scipy.constants.c
        )

        ## Basic path loss for LoS
        if distance_2d < distance_breakpoint:
            pl_los = (
                20.0 * np.log10(40.0 * np.pi * distance_3d * fc_ghz / 3.0)
                + np.min([0.03 * np.power(average_building_height, 1.72), 10.0])
                * np.log10(distance_3d)
                - np.min([0.044 * np.power(average_building_height, 1.72), 14.77])
                + 0.002 * np.log10(average_building_height) * distance_3d
            )
        else:
            pl_los = (
                20.0 * np.log10(40.0 * np.pi * distance_breakpoint * fc_ghz / 3.0)
                + np.min([0.03 * np.power(average_building_height, 1.72), 10.0])
                * np.log10(distance_breakpoint)
                - np.min([0.044 * np.power(average_building_height, 1.72), 14.77])
                + 0.002 * np.log10(average_building_height) * distance_breakpoint
                + 40.0 * np.log10(distance_3d / distance_breakpoint)
            )

        ## Basic pathloss for NLoS and O2I
        if not los:
            pl_3 = (
                161.04
                - 7.1 * np.log10(average_street_width)
                + 7.5 * np.log10(average_building_height)
                - (24.37 - 3.7 * np.square(average_building_height / h_bs))
                * np.log10(h_bs)
                + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(distance_3d) - 3.0)
                + 20.0 * np.log10(fc_ghz)
                - (3.2 * np.square(np.log10(11.75 * h_ut)) - 4.97)
            )
            pl_los = np.max([pl_los, pl_3])

        return pl_los

    def urban_path_loss(self, distance_2d, distance_3d, fc_ghz, h_ut, los):
        h_bs = 25

        # # LOS condition
        # if distance_2d < 18.0:
        #     los_probability = 1.0
        # else:
        #     c = 0 if h_ut < 13.0 else np.power((h_ut-13.)/10., 1.5)
        #     los_probability = ((18.0/distance_2d
        #         + np.exp(-distance_2d/63.0)*(1.-18./distance_2d))
        #         *(1.+c*5./4.*np.power(distance_2d/100., 3)
        #             *np.exp(-distance_2d/150.0)))
        # los = np.random.binomial(1, los_probability) # Bernoulli Distribution

        # Beak point distance
        g = (
            (5.0 / 4.0)
            * np.power(distance_2d / 100.0, 3.0)
            * np.exp(-distance_2d / 150.0)
        )
        g = g if distance_2d >= 18.0 else 0.0
        c = 0.0 if h_ut < 13.0 else g * np.power((h_ut - 13.0) / 10.0, 1.5)
        p = 1.0 / (1.0 + c)
        r = np.random.uniform()
        r = 1.0 if r < p else 0.0

        max_value = h_ut - 1.5
        s = np.random.uniform(12, max_value)
        # It could happen that h_ut = 13m, and therefore max_value < 13m
        s = s if s >= 12.0 else 12.0

        h_e = r + (1.0 - r) * s
        h_bs_prime = h_bs - h_e
        h_ut_prime = h_ut - h_e
        # For this computation, the carrifer frequency needs to be in Hz
        distance_breakpoint = (
            4 * h_bs_prime * h_ut_prime * fc_ghz * 1e9 / scipy.constants.c
        )

        ## Basic path loss for LoS
        if distance_2d < distance_breakpoint:
            pl = 28.0 + 22.0 * np.log10(distance_3d) + 20.0 * np.log10(fc_ghz)
        else:
            pl = (
                28.0
                + 40.0 * np.log10(distance_3d)
                + 20.0 * np.log10(fc_ghz)
                - 9.0
                * np.log10(np.square(distance_breakpoint) + np.square(h_bs - h_ut))
            )

        ## Basic pathloss for NLoS
        if not los:
            pl_3 = (
                13.54
                + 39.08 * np.log10(distance_3d)
                + 20.0 * np.log10(fc_ghz)
                - 0.6 * (h_ut - 1.5)
            )
            pl = np.max([pl, pl_3])

        return pl

    def calc_scenario_pl(self, path_data, fc=2.45, h_ut=5, los=False):
        total = 0
        for pdata in path_data:
            # TODO: Confirm [-1] and fc, h_ut and los defaults
            distances = pdata["distances"]*self.mpp
            cumsum_d = np.cumsum(distances)
            for d in cumsum_d:
                # urban_pl = self.urban_path_loss(d, d, fc, h_ut, los).flatten()
                rural_pl = self.rural_path_loss(d, d, fc, h_ut, los).flatten()
                # extra_urban_pl = np.diff(urban_pl)
                # extra_rural_pl = np.diff(rural_pl)

                # pl = urban_pl[0] if pdata["terrain_type"][0] > 0 else rural_pl[0]
                pl = rural_pl[0]
                # for i in range(len(extra_urban_pl)):
                #     if pdata["terrain_type"][i + 1] > 0:
                #         pl += extra_urban_pl[i]
                #     else:
                #         pl += extra_rural_pl[i]
                total += pl
            ...

        return total

    def calc_scenario_received_power(self, path_data):
        return -self.calc_scenario_pl(path_data)


if __name__ == "__main__":
    seed = 163250513
    np.random.seed(seed)
    random.seed(seed)
    print("Generating map")
    scen_gen = DisAMRScenarioGenerator(seed=seed, n_receivers=10)
    scen_gen.PlotMap(save_path="C_.png")
