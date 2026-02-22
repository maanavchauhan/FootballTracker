"""
Ball detection entity.

Handles isolation and padding of ball bounding boxes.
"""

import supervision as sv  # type: ignore

from entity.detections import Detections


class Ball(Detections):
    """
    Handles ball detection logic (padding, isolation).
    """

    def __init__(self, frame, detections, tracker, team_classifier, **kwargs):
        super().__init__(frame, detections, tracker, team_classifier, **kwargs)

    def pad_ball_detections(self, ball_detections: sv.Detections) -> sv.Detections:
        """
        Pads bounding boxes around the ball.
        """
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=self.pad_px)
        return ball_detections

    def process(self) -> sv.Detections:
        """
        Extracts and pads ball detections.
        """
        ball = self.get_class_detections(self.BALL_ID)
        return self.pad_ball_detections(ball)
