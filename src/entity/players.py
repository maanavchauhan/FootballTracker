"""
Player detection entity.

Handles NMS, tracking, and team classification for outfield players.
"""

import io
import contextlib

import supervision as sv  # type: ignore

from entity.detections import Detections


class Players(Detections):
    """
    Handles player detection refinement, tracking, and team classification.
    """

    def __init__(self, frame, detections, tracker, team_classifier, **kwargs):
        super().__init__(frame, detections, tracker, team_classifier, **kwargs)

    def apply_nms_and_track(self):
        """
        Applies Non-Maximum Suppression and updates tracker for all non-ball detections.
        """
        detections = self.detections[self.detections.class_id != self.BALL_ID]
        detections = detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
        return self.tracker.update_with_detections(detections)

    def classify_teams(self, player: sv.Detections) -> sv.Detections:
        """
        Uses the team classifier to assign team IDs to player detections.
        """
        player_crops = [sv.crop_image(self.frame, xyxy) for xyxy in player.xyxy]
        with contextlib.redirect_stderr(io.StringIO()):
            player.class_id = self.team_classifier.predict(player_crops)
        return player

    def process(self):
        """
        Executes full pipeline: NMS, tracking, and team classification.
        """
        tracked = self.apply_nms_and_track()
        players = tracked[tracked.class_id == self.PLAYER_ID]
        players = self.classify_teams(players)
        return tracked, players
