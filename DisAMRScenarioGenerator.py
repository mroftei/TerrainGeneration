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


class TerrainType(IntEnum):
    Urban = 2
    Rural = 1


class Direction(IntEnum):
    Up = 2
    Down = 3
    Left = 4
    Right = 5
    Towards = 1
    Away = -1
    Clockwise = 7
    CounterClockwise = 8


class DisAMRScenarioGenerator:
    def __init__(
        self,
        n_receivers=1,
        resolution=1000,
        min_receiver_dist=200,
        min_path_distance=10,
        seed=42,
        meters_per_pixel=10,
        target_path_loss=150,
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

        self.RegenerateFullScenario(target_path_loss)

    def RegenerateFullScenario(
        self,
        target_path_loss,
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
            self.transmitters, self.receivers, target_path_loss
        )

    def create_nodes(self, senders_in_city, map, map_resolution):
        if senders_in_city:
            city_points = np.stack(np.where(map == TerrainType.Urban)).T
            transmitters = city_points[
                self.rng.integers(len(city_points), size=(self.n_tx)), :
            ]
            transmitters[:, [0, 1]] = transmitters[
                :, [1, 0]
            ]  # Swap colums for consistency during unpack
        else:
            transmitters = self.rng.integers(map_resolution, size=(self.n_tx, 2))
        receivers = self.rng.integers(map_resolution, size=(self.n_rx, 2))

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
        clipped_new_coord = np.clip(new_coord, 0, self.resolution - 1)
        if (new_coord != clipped_new_coord).any():
            clip = True
            new_coord = clipped_new_coord
        else:
            clip = False

        return new_coord, clip
    
    def get_path_data(self):
        return self._computeDistances(self.transmitters, self.receivers, self.map)

    def RegenerateReceivers(
        self,
        transmitters,
        receivers,
        target_path_loss,
        error=0.01,
        initial_magnitude=1,
        iteration_max=200,
        save_metrics=False,
    ):
        dist = cdist(transmitters, receivers, "euclidean").min()
        while dist < self.min_receiver_dist:
            receivers = self.rng.integers(0, self.resolution, size=(self.n_rx, 2))
            dist = cdist(transmitters, receivers, "euclidean").min()
        if target_path_loss == inf:
            return receivers

        path_data = self._computeDistances(transmitters, receivers, self.map)
        path_loss = self.calc_scenario_pl(path_data)
        iteration_idx = 0
        reciever_lengths = []
        magnitude = initial_magnitude
        init_sender = [0]*len(path_data)
        init_receiver = [0]*len(path_data)
        for i in range(len(path_data)):
            init_sender[i], init_receiver[i] = (
                path_data[i]["sender_coords"].copy(),
                path_data[i]["receiver_coords"].copy(),
            )            
        
        leap = 1.5
        decay = 3
        for i in range(len(path_data)):
            magnitude = initial_magnitude
            iteration_idx = 0
            min_diff = inf
            while abs(path_loss - target_path_loss) > error and iteration_idx < iteration_max:
                sender, receiver = (
                    path_data[i]["sender_coords"],
                    path_data[i]["receiver_coords"],
                )
                path_loss = self.calc_scenario_pl(path_data)
                diff = path_loss - target_path_loss
                direction = Direction.Towards if diff > 0 else Direction.Away
                dist = cdist([sender], [receiver], "euclidean").item()
                if direction == Direction.Towards:
                    magnitude = np.clip(magnitude, 0, dist) + self.min_receiver_dist

                # print(f"PRE MOVE: RECEIVER {i}, ITERATION {iteration_idx}, distance: {dist}, error: {diff}, magnitude: {magnitude}, path_loss: {path_loss}, Direction: {str(direction)}")
                receiver, clip = self.move(
                    init_receiver[i], direction, magnitude=magnitude, relative_to_coord=init_sender[i]
                )
                receiver_data = next(
                    iter(
                        self._computeDistances(
                            [sender], [receiver], self.map, path_aggregation=True
                        )
                    )
                )
                tmp_path_data = deepcopy(path_data)
                tmp_path_data[i] = receiver_data
                path_loss = self.calc_scenario_pl(tmp_path_data)
                final_diff = path_loss - target_path_loss
                dist = cdist([sender], [receiver], "euclidean").item()
                # print(f"POST MOVE: RECEIVER {i}, ITERATION {iteration_idx}, distance: {dist}, error: {final_diff}, magnitude: {magnitude}, path_loss: {path_loss}, Direction: {str(direction)}")

                if abs(final_diff) < min_diff:
                    optimal_path_data = deepcopy(tmp_path_data)
                    min_diff = abs(final_diff)                
                if np.abs(dist - self.min_receiver_dist) < 0.000001 or clip:
                    break
                if np.sign(final_diff)[0] == np.sign(diff)[0]:
                    magnitude *= leap * np.exp(-iteration_idx/iteration_max)
                    # If you need more magnitude, and moving away, and you clipped then stop
                    if direction == Direction.Away and clip:
                        break
                    # If you need more magnitude, and moving towards, and you're already basically there then stop
                    if direction == Direction.Towards and np.abs(dist - self.min_receiver_dist) < 0.000001:
                        break
                else:
                    magnitude /= leap * np.exp(-decay*iteration_idx/iteration_max)
                
                iteration_idx += 1
            path_data = deepcopy(optimal_path_data)
        
        if save_metrics:
            pd.DataFrame.from_records(reciever_lengths).to_csv("regen_rx_placement.csv")

        receivers = [p["receiver_coords"] for p in optimal_path_data]

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
            num = int(np.linalg.norm(map.shape)*10)
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
            terrain_types = np.array(terrain_types).T
            key_points = np.stack([xtrans_coords, ytrans_coords], axis=1)
            point_pairs = sliding_window_view(key_points, window_shape=(2, 2)).squeeze()
            if point_pairs.shape == (2, 2):
                point_pairs = point_pairs[np.newaxis, ...]
            distances = np.array(
                [pdist(pair, metric=distance_metric) for pair in point_pairs]
            ).T
            new_path = {
                "key_points": key_points,
                "receiver_coords": receiver,
                "sender_coords": sender,
                "terrain_type": terrain_types,
                "distances": distances,
                "distances_m": distances * self.mpp,
                "receiver_coords_m": receiver * self.mpp,
                "sender_coords_m": sender * self.mpp,
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
            point_pairsa_temp = []
            terrain_temp = []
            i = 0
            while i < len(new_point_pairs):
                j = i + 1
                while j < len(new_point_pairs) and terrain_types[i] == terrain_types[j]:
                    j += 1
                point_pairsa_temp.append(
                    np.stack((new_point_pairs[i][0], new_point_pairs[j - 1][-1]))
                )
                terrain_temp.append(terrain_types[j - 1])
                i = j
                
            new_point_pairs = np.stack(point_pairsa_temp)
            terrain_types = np.stack(terrain_temp)

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
            ).T
            terrain_types = np.array(terrain_types).T
            new_path = {
                "key_points": key_points,
                "receiver_coords": receiver,
                "sender_coords": sender,
                "terrain_type": terrain_types,
                "distances": distances,
                "distances_m": distances * self.mpp,
                "receiver_coords_m": receiver * self.mpp,
                "sender_coords_m": sender * self.mpp,
            }
            new_distances = distances.sum()
            if abs(old_distances-new_distances) > 0.01:
                raise Exception
            path_data.append(new_path)
        return path_data

    def rural_path_loss(self, distance_2d, distance_3d, fc_ghz, h_ut, los=False):
        h_bs = 25
        average_building_height = 5.0 #5:50
        average_street_width = 20 #5:50

        # Beak point distance
        # For this computation, the carrifer frequency needs to be in Hz
        distance_breakpoint = (2.*scipy.constants.pi*h_bs*h_ut*fc_ghz*1e9/scipy.constants.c)

        ## Basic path loss for LoS
        pl_short = (20.0*np.log10(40.0*np.pi*distance_3d*fc_ghz/3.)
            + np.min([0.03*np.power(average_building_height,1.72), 10.0])*np.log10(distance_3d)
            - np.min([0.044*np.power(average_building_height,1.72), 14.77])
            + 0.002*np.log10(average_building_height)*distance_3d)
        pl_long = (20.0*np.log10(40.0*np.pi*distance_breakpoint*fc_ghz/3.)
            + np.min([0.03*np.power(average_building_height,1.72), 10.0])*np.log10(distance_breakpoint)
            - np.min([0.044*np.power(average_building_height,1.72), 14.77])
            + 0.002*np.log10(average_building_height)*distance_breakpoint
            + 40.0*np.log10(distance_3d/distance_breakpoint))
        pl = np.where(distance_2d < distance_breakpoint, pl_short, pl_long)

        ## Basic pathloss for NLoS
        pl_3 = (161.04 - 7.1*np.log10(average_street_width)
                + 7.5*np.log10(average_building_height)
                - (24.37 - 3.7*np.square(average_building_height/h_bs))*np.log10(h_bs)
                + (43.42 - 3.1*np.log10(h_bs))*(np.log10(distance_3d)-3.0)
                + 20.0*np.log10(fc_ghz) - (3.2*np.square(np.log10(11.75*h_ut))
                - 4.97))
        idx = np.logical_not(los)
        pl = np.where(idx, np.max([pl, pl_3], 0), pl)

        return pl

    def urban_path_loss(self, distance_2d, distance_3d, fc_ghz, h_ut, los=False):
        h_bs = 25
        
        # Beak point distance
        g = ((5./4.)*np.power(distance_2d/100., 3.)
            *np.exp(-distance_2d/150.0))
        g[distance_2d < 18.0] = 0.0
        c = np.zeros_like(g)
        c_idx = h_ut >= 13.0
        if np.any(c_idx):
            c[c_idx] = g[c_idx]*np.power((h_ut[c_idx]-13.)/10., 1.5)
        p = 1./(1.+c)
        r = self.rng.uniform(size=distance_2d.shape)
        r = np.where(r<p, 1.0, 0.0)

        max_value = h_ut - 1.5
        max_value = np.clip(max_value, 12.0, None)
        s = self.rng.uniform(12, max_value, size=(len(distance_2d),1))
        # It could happen that h_ut = 13m, and therefore max_value < 13m

        h_e = r + (1.-r)*s
        h_bs_prime = h_bs - h_e
        h_ut_prime = h_ut - h_e
        # For this computation, the carrifer frequency needs to be in Hz
        distance_breakpoint = 4*h_bs_prime*h_ut_prime*fc_ghz*1e9/scipy.constants.c

        ## Basic path loss for LoS
        pl_short = 28.0 + 22.0*np.log10(distance_3d) + 20.0*np.log10(fc_ghz)
        pl_long = (28.0 + 40.0*np.log10(distance_3d) + 20.0*np.log10(fc_ghz)
            - 9.0*np.log10(np.square(distance_breakpoint)+np.square(h_bs-h_ut)))
        pl = np.where(distance_2d < distance_breakpoint, pl_short, pl_long)

        ## Basic pathloss for NLoS
        pl_3 = (13.54 + 39.08*np.log10(distance_3d) + 20.0*np.log10(fc_ghz)
            - 0.6*(h_ut-1.5))
        idx = np.logical_not(los)
        pl = np.where(idx, np.max([pl, pl_3], 0), pl)

        return pl


    def calc_scenario_pl(self, path_data, fc_ghz=0.92, h_ut=1.5, los=True):
        total_pl = 0 
        for pData in path_data:
            cumsum_d = np.cumsum(pData["distances_m"], axis=-1)
            urban_pl_dB = self.urban_path_loss(cumsum_d, cumsum_d, fc_ghz, h_ut, los)
            rural_pl_dB = self.rural_path_loss(cumsum_d, cumsum_d, fc_ghz, h_ut, los)
            urban_pl_lin = 10**(urban_pl_dB/10)
            rural_pl_lin = 10**(rural_pl_dB/10)
            extra_urban_pl = np.diff(urban_pl_lin)
            extra_rural_pl = np.diff(rural_pl_lin)
            #TODO: log math
            
            pl = np.where(pData["terrain_type"][...,0] > 0, urban_pl_lin[...,0], rural_pl_lin[...,0])
            extra_pl = np.where(pData["terrain_type"][...,1:] > 0, extra_urban_pl, extra_rural_pl)
            pl += np.sum(extra_pl, axis=-1)
            total_pl += pl
        return 10*np.log10(total_pl)


if __name__ == "__main__":
    seed = 1052313
    np.random.seed(seed)
    random.seed(seed)
    print("Generating map")
    scen_gen = DisAMRScenarioGenerator(seed=seed, n_receivers=10)
    scen_gen.PlotMap(save_path="C_.png")
