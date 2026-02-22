"""
Frame-level detection orchestrator.

Combines ball, player, goalkeeper, and referee detections into a single
coherent result set for each video frame.
"""

import supervision as sv  # type: ignore

from entity.players import Players
from entity.goalkeepers import Goalkeepers
from entity.ball import Ball


class FrameDetections:
    """
    Orchestrates detection processing for each frame — combining
    ball, player, goalkeeper, and referee detections.
    """

    def __init__(self, frame, detections, tracker, team_classifier, **kwargs):
        self.frame = frame
        self.detections = detections
        self.tracker = tracker
        self.team_classifier = team_classifier

        # ---- Ball ----
        self.ball_handler = Ball(frame, detections, tracker, team_classifier, **kwargs)
        self.ball_detections = self.ball_handler.process()

        # ---- Players ----
        self.player_handler = Players(frame, detections, tracker, team_classifier, **kwargs)
        all_detections, self.players_detections = self.player_handler.process()

        # ---- Goalkeepers ----
        self.goalkeeper_handler = Goalkeepers(frame, detections, tracker, team_classifier, **kwargs)
        self.goalkeepers_detections = self.goalkeeper_handler.process(all_detections, self.players_detections)

        # ---- Referees ----
        self.referees_detections = all_detections[all_detections.class_id == self.player_handler.REFEREE_ID]
        self.referees_detections.class_id -= 1

        # ---- Combine results ----
        self.all_players_detections = sv.Detections.merge([
            self.players_detections, self.goalkeepers_detections
        ])

        self.all_detections = sv.Detections.merge([
            self.players_detections, self.goalkeepers_detections, self.referees_detections
        ])

        self.labels = [f"#{tid}" for tid in self.all_detections.tracker_id]
        self.all_detections.class_id = self.all_detections.class_id.astype(int)
