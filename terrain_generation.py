import numpy as np
import matplotlib.pyplot as plt
import opensimplex
import random
from scipy.spatial.distance import pdist
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product
import matplotlib.patches as mpatches
from enum import IntEnum
import json
from itertools import groupby
from scipy.stats import mode
from matplotlib.lines import Line2D

SEED = 1632505


class TerrainType(IntEnum):
    Open = 1
    Foliage = 2
    Suburban = 3
    Urban = 4


def create_map(
    freq=1,
    map_resolution=100,
    octaves=1,
    octave_factor=2,
    redistribution=1,
    levels=3,
    island=False,
):
    # Initialize map
    z = np.zeros((map_resolution, map_resolution))

    # Apply noise maps with different frequencies
    of = 1
    for _ in range(octaves):
        x, y = np.linspace(freq * of * -1, freq * of * 1, map_resolution), np.linspace(
            freq * of * -1, freq * of * 1, map_resolution
        )
        z += (1 / of) * opensimplex.noise2array(x, y)
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


def resolve_map(pop_map, foliage_map):
    assert pop_map.shape == foliage_map.shape
    map = np.ones(pop_map.shape)
    # map[pop_map == 0] = float(TerrainType.Open)  # Terrain
    map[np.where(foliage_map> 0.7)] = float(TerrainType.Foliage)  # Foliage
    # map[np.where((pop_map >= 0.8) & (foliage_map <= 0.5))] = float(TerrainType.Urban)  # Urban
    map[np.where(pop_map >= 0.7)] = float(TerrainType.Suburban)
    map[np.where(pop_map >= 0.9)] = float(TerrainType.Urban)
    return map


def create_nodes(senders, recievers, sender_in_city, map, map_resolution):
    if sender_in_city:
        city_points = np.stack(np.where(map == TerrainType.Urban)).T
        senders = city_points[np.random.randint(len(city_points), size=(senders)), :]
        senders[:, [0, 1]] = senders[
            :, [1, 0]
        ]  # Swap colums for consistency during unpack
    else:
        senders = np.random.randint(map_resolution, size=(senders, 2))
    recievers = np.random.randint(map_resolution, size=(recievers, 2))
    return senders, recievers


def get_consecutive_group(d):
    v_last = d[0]
    group = [v_last]
    v_idx = 1
    while v_idx < len(d):
        v = d[v_idx]
        if (v - 1) != v_last:
            yield (group)
            v_last = d[v_idx]
            group = [v_last]
            v_idx += 1
        else:
            group.append(v)
            v_last = v
            v_idx += 1
    yield group


def compute_distances(
    senders,
    recievers,
    map,
    distance_metric="euclidean",
    path_aggregation=True,
    distance_aggregation_threshold=0.001,
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
        terrain_types = map[ytrans_coords.astype(int), xtrans_coords.astype(int)][:-1]
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
        key_points = np.unique(
            np.concatenate(
                [
                    sender.reshape(-1, 2),
                    new_point_pairs.reshape(-1, 2),
                    reciever.reshape(-1, 2),
                ],
                axis=0,
            ),
            axis=0,
        )
        point_pairs = sliding_window_view(key_points, window_shape=(2, 2)).squeeze()
        distances = np.array(
            [pdist(pair, metric=distance_metric) for pair in new_point_pairs]
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
    return path_data


def plot_map(map, path_data, show_signal_paths, save_path):
    fig, axes = plt.subplots(nrows=1, figsize=(10, 10))
    im = axes.imshow(map, origin="lower", cmap="Blues")
    values = np.unique(map.ravel())
    if not show_signal_paths:
        plt.show()
        return

    for d in path_data:
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
            color=colors[v_idx], label=f"{str(TerrainType(v)).split('.')[-1]} Terrain"
        )
        for v_idx, v in enumerate(values)
    ]
    patches.extend([Line2D([], [], label="Reciever", color="white", marker='o', markerfacecolor="g")])
    patches.extend([Line2D([], [], label="Sender", color="white", marker='o', markerfacecolor="y")])
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.grid(True)

    if len(save_path):
        fig.savefig(save_path, bbox_inches="tight")
        return fig

    plt.show()
    return fig


def generate_map(
    senders=1,
    receivers=10,
    show_signal_paths=True,
    sender_in_city=True,
    plot=False,
    seed=SEED,
    save_path="",
    return_map_data=True,
    distance_aggregation_threshold=0.0187,
    **map_config,
):
    """Used to generation sender/ reciever maps with support for different terrain types

    Args:
        senders (int, optional): Sender count. Defaults to 1.
        receivers (int, optional): Reciever count. Defaults to 10.
        show_signal_paths (bool, optional): Enable plotting of signal paths from senders to recievers. Defaults to True.
        highlight_senders (bool, optional): Distinguish senders when plotting. Defaults to True.
        sender_in_city (bool, optional): Constraint sender location within city terrain boundaries. Defaults to True.
        plot (bool, optional): Enable plotting. Defaults to False.
        seed (int, optional): Seed used for rngs. Defaults to 163250507.
        save_path (str, optional): If set disables interactive plotting and saves plot to filesystem. Defaults to "".
        return_map_data (bool, optional): If set to True returns map and distance data. Defaults to True.

    Returns:
        List[Dict[str, np.array]]: Ordered list of records with data on all paths from all senders to all recievers Data includes the following:
            key_points: Ordered list of coordinates for every terrain transition starting from sender to reciever,
            reciever_coords: xy coordinates of reciever,
            sender_coords: xy coordinates of sender,
            terrain_type: Ordered list of different terrain types encountered from sender to reciever,
            distances: Ordered list of distances between key_points marking terrain transitions from sender to reciever,

    """
    opensimplex.seed(seed)

    pop_map = create_map(
        map_config["pop_freq"],
        map_config["map_resolution"],
        map_config["pop_octaves"],
        map_config["pop_octaves_factor"],
        map_config["pop_redistribution"],
        map_config["pop_levels"],
        island=True,
    )
    foliage_map = create_map(
        map_config["foliage_freq"],
        map_config["map_resolution"],
        map_config["foliage_octaves"],
        map_config["foliage_octaves_factor"],
        map_config["foliage_redistribution"],
        map_config["foliage_levels"],
    )
    map = resolve_map(pop_map, foliage_map)

    senders, recievers = create_nodes(
        senders, receivers, sender_in_city, map, map_config["map_resolution"]
    )
    path_data = compute_distances(
        senders,
        recievers,
        map,
        distance_aggregation_threshold=distance_aggregation_threshold,
    )
    if plot:
        plot_map(map, path_data, show_signal_paths, save_path)

    if not return_map_data:
        return

    return map, path_data, senders, receivers


if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    print("Generating map")
    default_map_config = json.load(open("default_map_config.json"))
    generate_map(
        sender_in_city=False, senders=10, receivers=2, plot=True, return_map_data=True, save_path="multi_sender.png", **default_map_config
    )
    print("Done")
