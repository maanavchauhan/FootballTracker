"""
Goalkeeper detection entity.

Assigns each goalkeeper to a team by proximity to player centroids.
"""

import numpy as np
import supervision as sv  # type: ignore

from entity.detections import Detections


class Goalkeepers(Detections):
    """
    Handles goalkeeper detection and team assignment logic.
    """

    def __init__(self, frame, detections, tracker, team_classifier, **kwargs):
        super().__init__(frame, detections, tracker, team_classifier, **kwargs)

    def resolve_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        """
        Determines which team each goalkeeper belongs to by comparing distances
        to player team centroids.
        """
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)

        team_ids = []
        for gk_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
            dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
            team_ids.append(0 if dist_0 < dist_1 else 1)

        return np.array(team_ids)

    def process(self, all_detections: sv.Detections, player_detections: sv.Detections) -> sv.Detections:
        """
        Extracts goalkeepers and assigns their team IDs.
        """
        goalkeepers = all_detections[all_detections.class_id == self.GOALKEEPER_ID]
        goalkeepers.class_id = self.resolve_team_id(player_detections, goalkeepers)
        return goalkeepers
