"""
Ball position tracking utilities.

Provides functions to extract, clean, and interpolate ball positions
across video frames for smoother tracking results.
"""

import numpy as np
import pandas as pd
import supervision as sv  # type: ignore

from typing import List, Union
from config import BALL_ID


def interpolate_ball_positions(ball_positions) -> List[np.ndarray]:
    """
    Fills missing ball positions via linear interpolation and back-fill.
    """
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()
    ball_positions = df_ball_positions.to_numpy().tolist()
    return ball_positions


def replace_outliers_based_on_distance(
    positions: List[np.ndarray],
    distance_threshold: float,
) -> List[np.ndarray]:
    """
    Removes positions that jump further than *distance_threshold* from
    the last valid position, replacing them with empty arrays.
    """
    last_valid_position: Union[np.ndarray, None] = None
    cleaned_positions: List[np.ndarray] = []

    for position in positions:
        if len(position) == 0:
            cleaned_positions.append(position)
        else:
            if last_valid_position is None:
                cleaned_positions.append(position)
                last_valid_position = position
            else:
                distance = np.linalg.norm(position - last_valid_position)
                if distance > distance_threshold:
                    cleaned_positions.append(np.array([], dtype=np.float64))
                else:
                    cleaned_positions.append(position)
                    last_valid_position = position

    return cleaned_positions


def get_ball_positions(record):
    """
    Extracts per-frame ball bounding boxes from the detection record.

    When multiple balls are detected in a single frame, the one closest
    to the centroid of neighbouring frames is selected.
    """
    ball_positions = []
    num_frames = len(record)

    for i, frame_info in enumerate(record):
        detections = frame_info['frameDetections'].detections
        ball_detections = detections[detections.class_id == BALL_ID]
        index = 0

        if len(ball_detections) == 0:
            ball_positions.append(ball_detections.xyxy)
            continue
        elif len(ball_detections) > 1:
            # Resolve ambiguity using neighbouring frames
            xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
            neighbor_positions = []
            start = max(0, i - 5)
            end = min(num_frames, i + 6)

            for j in range(start, end):
                if j == i:
                    continue
                neighbor_frame = record[j]
                neighbor_detections = neighbor_frame['frameDetections'].detections
                neighbor_ball_detections = neighbor_detections[neighbor_detections.class_id == BALL_ID]
                if len(neighbor_ball_detections) > 0:
                    neighbor_xy = neighbor_ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                    if neighbor_xy.ndim > 1 and neighbor_xy.shape[0] > 1:
                        neighbor_xy = np.mean(neighbor_xy, axis=0)
                    neighbor_positions.append(neighbor_xy)

            if neighbor_positions:
                centroid = np.mean(np.array(neighbor_positions), axis=0)
                distances = np.linalg.norm(xy - centroid, axis=1)
                index = np.argmin(distances)

        ball_positions.append(ball_detections.xyxy[index])

    return ball_positions
