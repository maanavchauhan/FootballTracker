"""
Base detection class for the entity hierarchy.

All specific detection types (Ball, Players, Goalkeepers) inherit from
this class, sharing common attributes and filtering logic.
"""

import supervision as sv  # type: ignore

from config import (
    BALL_ID as DEFAULT_BALL_ID,
    GOALKEEPER_ID as DEFAULT_GOALKEEPER_ID,
    PLAYER_ID as DEFAULT_PLAYER_ID,
    REFEREE_ID as DEFAULT_REFEREE_ID,
)


class Detections:
    """
    Base class for all types of detections in a football frame.
    Holds shared attributes and methods common to all detection types.
    """

    def __init__(
        self,
        frame,
        detections,
        tracker,
        team_classifier,
        BALL_ID=DEFAULT_BALL_ID,
        GOALKEEPER_ID=DEFAULT_GOALKEEPER_ID,
        PLAYER_ID=DEFAULT_PLAYER_ID,
        REFEREE_ID=DEFAULT_REFEREE_ID,
        pad_px=10,
        nms_threshold=0.5,
    ):
        self.frame = frame
        self.detections = detections
        self.tracker = tracker
        self.team_classifier = team_classifier

        # Class IDs
        self.BALL_ID = BALL_ID
        self.GOALKEEPER_ID = GOALKEEPER_ID
        self.PLAYER_ID = PLAYER_ID
        self.REFEREE_ID = REFEREE_ID

        # Parameters
        self.pad_px = pad_px
        self.nms_threshold = nms_threshold

    def get_class_detections(self, class_id: int) -> sv.Detections:
        """
        Filters detections by a given class ID.
        """
        return self.detections[self.detections.class_id == class_id]
