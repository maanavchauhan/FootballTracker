"""
Offside detection logic.

Identifies the last defender line for each team and flags players
that are in a potential offside position.
"""

import numpy as np

from entity.frame_detections import FrameDetections
from entity.pitch_detections import PitchDetections


def find_last_defender(f: FrameDetections, p: PitchDetections):
    """
    Determines the last defender x-coordinate for each team and returns
    a list of tracker IDs for players in a potential offside position.

    Returns:
        last_defender (dict | None): {0: x_coord, 1: x_coord} for each team.
        potential_offside (list): Tracker IDs of potentially offside players.
    """
    last_defender = {}
    potential_offside = []

    if len(p.pitch_players_xy) > 0:
        if np.mean(p.pitch_team0_players_xy[:, 0]) < np.mean(p.pitch_team1_players_xy[:, 0]):
            last_defender[0] = np.min(p.pitch_team0_players_xy[:, 0])
            last_defender[1] = np.max(p.pitch_team1_players_xy[:, 0])

            for pos, tracker in zip(
                p.pitch_team0_players_xy,
                f.players_detections[f.players_detections.class_id == 0].tracker_id,
            ):
                if pos[0] > last_defender[1]:
                    potential_offside.append(tracker)

            for pos, tracker in zip(
                p.pitch_team1_players_xy,
                f.players_detections[f.players_detections.class_id == 1].tracker_id,
            ):
                if pos[0] < last_defender[0]:
                    potential_offside.append(tracker)
        else:
            last_defender[0] = np.max(p.pitch_team0_players_xy[:, 0])
            last_defender[1] = np.min(p.pitch_team1_players_xy[:, 0])

            for pos, tracker in zip(
                p.pitch_team0_players_xy,
                f.players_detections[f.players_detections.class_id == 0].tracker_id,
            ):
                if pos[0] < last_defender[1]:
                    potential_offside.append(tracker)

            for pos, tracker in zip(
                p.pitch_team1_players_xy,
                f.players_detections[f.players_detections.class_id == 1].tracker_id,
            ):
                if pos[0] > last_defender[0]:
                    potential_offside.append(tracker)
    else:
        last_defender = None

    return last_defender, potential_offside
