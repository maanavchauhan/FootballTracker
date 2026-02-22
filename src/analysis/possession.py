"""
Ball possession detection.

Determines which player (if any) currently possesses the ball based on
proximity to the ball's centre.
"""

import numpy as np
import supervision as sv  # type: ignore

from config import PLAYER_IN_POSSESSION_PROXIMITY
from entity.frame_detections import FrameDetections


def get_player_in_possession(f: FrameDetections, proximity=PLAYER_IN_POSSESSION_PROXIMITY):
    """
    Returns the tracker ID of the player closest to the ball, provided
    that player is within *proximity* pixels.  Returns -1 when no player
    qualifies or the ball is not uniquely detected.
    """
    if len(f.ball_detections) != 1:
        return -1

    ball_centers = f.ball_detections.get_anchors_coordinates(sv.Position.CENTER)
    ball_center = ball_centers[0]
    players_xy = f.players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    min_distance = np.linalg.norm(players_xy[0] - ball_center)
    player = -1

    for player_xy, player_id in zip(players_xy, f.players_detections.tracker_id):
        distance = np.linalg.norm(player_xy - ball_center)
        if distance < min_distance:
            min_distance = distance
            player = player_id

    if min_distance <= proximity:
        return player
    return -1
