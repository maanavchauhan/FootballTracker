"""
Player distance and speed tracking.

Computes cumulative distance covered and instantaneous speed for every
tracked player across frames.
"""

import numpy as np

from config import FPS, X_SCALE, Y_SCALE


def compute_distances_and_speeds(record, sample_interval=5):
    """
    Iterates over the frame *record* and, at every *sample_interval*
    frames, updates cumulative distances and instantaneous speeds for
    each tracked player.

    The results are written back into each frame_info dict as
    'cumulative_distances' and 'instantaneous_speed'.
    """
    last_positions = {}
    cumulative_distances = {}
    instantaneous_speed = {}

    for frame_idx, frame_info in enumerate(record):
        if frame_idx % sample_interval == 0:
            pitch_det = frame_info['pitchDetections']

            for tracker_id, pos in zip(
                frame_info['frameDetections'].all_players_detections.tracker_id,
                pitch_det.pitch_all_players_xy,
            ):
                current_pos = np.array(pos)

                if tracker_id in last_positions:
                    last_pos = last_positions[tracker_id]
                    # Convert pitch units to metres
                    current_pos_m = np.array([current_pos[0] * X_SCALE, current_pos[1] * Y_SCALE])
                    last_pos_m = np.array([last_pos[0] * X_SCALE, last_pos[1] * Y_SCALE])

                    dist = np.linalg.norm(current_pos_m - last_pos_m)
                    cumulative_distances[tracker_id] = cumulative_distances.get(tracker_id, 0) + dist

                    # Speed in km/h
                    dt = sample_interval / FPS
                    speed_m_s = dist / dt
                    speed_km_h = speed_m_s * 3.6
                    instantaneous_speed[tracker_id] = speed_km_h
                else:
                    instantaneous_speed[tracker_id] = 0

                last_positions[tracker_id] = current_pos

        # Snapshot current state into the record
        frame_info['cumulative_distances'] = cumulative_distances.copy()
        frame_info['instantaneous_speed'] = instantaneous_speed.copy()
