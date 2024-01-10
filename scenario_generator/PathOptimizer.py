from enum import IntEnum
import pandas as pd
import numpy as np
from sionna_torch import SionnaScenario

class Direction(IntEnum):
    Up = 1
    Down = 2
    Left = 3
    Right = 4
    Towards = 5
    Away = 6
    Clockwise = 7
    CounterClockwise = 8

def _move(self, from_coord, direction, relative_to_coord=None, magnitude=1):
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
            hyp_c = np.sqrt((y_target - y) ** 2 + (x_target - x) ** 2) + eps
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
    # new_coord = np.clip(new_coord, 0, self.resolution - 1)

    return new_coord

def OptimizeReceivers(
    self,
    tx_xy,
    rx_xy,
    ter_map,
    map_res,
    target_path_loss = None,
    f_c = 900e6,
    bw = 30e3,
    error=0.01,
    iteration_max=1000,
    save_metrics=True,
):
    iteration_idx = 0
    reciever_lengths = []

    scenario = SionnaScenario(self.transmitters[None], self.receivers[None], self.map, f_c=f_c, bw=bw, seed=self.seed)
    path_loss = scenario.basic_pathloss
    diff = path_loss - target_path_loss

    direction = Direction.Towards if diff < 0 else Direction.Away
    while abs(diff) > error and iteration_idx < iteration_max:
        # magnitude = initial_magnitude * np.exp(-iteration_idx / (iteration_max/4))
        # magnitude = initial_magnitude * 1/(1+np.exp((iteration_idx-iteration_idx/2)/(iteration_max/20))) # Sigmoid descent
        magnitude = np.abs(diff)/100
        d = {}
        d["magnitude"] = magnitude
        d["error"] = abs(diff)
        for i in range(len(path_data)):
            sender, receiver = (
                path_data[i]["sender_coords"],
                path_data[i]["receiver_coords"],
            )
            receiver = self.move(
                receiver, direction, magnitude=magnitude, relative_to_coord=sender
            )
            dist = cdist([sender], [receiver], "euclidean").item()
            if dist < self.min_receiver_dist:
                d[f"receiver_{i}"] = path_data[i]["distances"].sum().item()
                d[f"direction_{i}"] = 1 if direction == Direction.Away else -1
                continue
            receiver_data = next(
                iter(
                    self._computeDistances(
                        [sender], [receiver], self.map, path_aggregation=True
                    )
                )
            )
            d[f"receiver_{i}"] = path_data[i]["distances"].sum().item()
            d[f"direction_{i}"] = 1 if direction == Direction.Away else -1
            path_data[i] = receiver_data

        path_loss = self.ChannelGain(path_data)
        diff = path_loss - target_path_loss
        direction = Direction.Towards if diff < 0 else Direction.Away

        d["P_rx"] = path_loss
        d["P_diff"] = diff
        reciever_lengths.append(d)
        receivers = [p["receiver_coords"] for p in path_data]
        iteration_idx += 1

    d["magnitude"] = magnitude
    d["error"] = abs(path_loss - target_path_loss)

    if save_metrics:
        pd.DataFrame.from_records(reciever_lengths).to_csv("regen_rx_placement.csv")

    return receivers, diff