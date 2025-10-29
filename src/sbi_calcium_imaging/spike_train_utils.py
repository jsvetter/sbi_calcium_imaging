from typing import Tuple

import numpy as np
import scipy.io as sio

# NOTE: many things taken from Peter Ruprechts repo


def load_ground_truth_mat(mat_path: str, recording_id: int = 0):
    """Load a single-neuron ground-truth recording from .mat files."""
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    CAttached = data["CAttached"]

    if isinstance(CAttached, np.ndarray):
        rec = CAttached[recording_id]
    else:
        rec = CAttached

    fluo_time = np.asarray(rec.fluo_time).ravel()
    fluo_mean = np.asarray(rec.fluo_mean).ravel()
    events_AP = np.asarray(rec.events_AP).ravel()
    ap_times_s = events_AP / 1e4

    return fluo_time, fluo_mean, ap_times_s


def make_spike_train(
    fluo_time: np.ndarray, ap_times_s: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Bin spike times to imaging time base."""
    dt = np.median(np.diff(fluo_time))
    if dt <= 0:
        raise ValueError("Non-increasing or invalid fluo_time.")

    edges = np.concatenate(
        [
            [fluo_time[0] - 0.5 * dt],
            0.5 * (fluo_time[1:] + fluo_time[:-1]),
            [fluo_time[-1] + 0.5 * dt],
        ]
    )

    spike_counts, _ = np.histogram(ap_times_s, bins=edges)
    spike_train = spike_counts.astype(float)

    return spike_train, float(dt)
