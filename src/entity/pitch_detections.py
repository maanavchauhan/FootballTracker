"""
Pitch-level detection coordinates.

Projects frame-space detections onto a 2-D pitch model using a
homography estimated from detected key points.
"""

import numpy as np
import supervision as sv  # type: ignore

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

from entity.frame_detections import FrameDetections


class PitchDetections:
    """
    Transforms frame-space coordinates to pitch-space for all detection
    categories (ball, players, goalkeepers, referees).
    """

    def __init__(self, frame, f: FrameDetections, key_points,
                 BALL_ID=0, GOALKEEPER_ID=1, PLAYER_ID=2, REFEREE_ID=3):
        self.f = f
        self.key_points = key_points
        self.BALL_ID = BALL_ID
        self.GOALKEEPER_ID = GOALKEEPER_ID
        self.PLAYER_ID = PLAYER_ID
        self.REFEREE_ID = REFEREE_ID
        CONFIG = SoccerPitchConfiguration()

        # Project key points from frame to pitch
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        self.transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

        self.transformer_inverse = ViewTransformer(
            source=pitch_reference_points,
            target=frame_reference_points
        )

        # ── Ball coordinates ────────────────────────────────────────────
        self.frame_ball_xy = f.ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self.pitch_ball_xy = self.transformer.transform_points(points=self.frame_ball_xy)

        # ── All players (outfield + goalkeepers) ────────────────────────
        self.all_players_xy = f.all_players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self.pitch_all_players_xy = self.transformer.transform_points(points=self.all_players_xy)

        # ── Outfield players only ───────────────────────────────────────
        self.players_xy = f.players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self.pitch_players_xy = self.transformer.transform_points(points=self.players_xy)

        # ── Referees ────────────────────────────────────────────────────
        self.referees_xy = f.referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self.pitch_referees_xy = self.transformer.transform_points(points=self.referees_xy)

        # ── Per-team player coordinates ─────────────────────────────────
        self.team_players_0 = f.players_detections[(f.players_detections.class_id == 0)]
        self.team_players_0_xy = self.team_players_0.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self.team_players_1 = f.players_detections[(f.players_detections.class_id == 1)]
        self.team_players_1_xy = self.team_players_1.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self.pitch_team0_players_xy = self.transformer.transform_points(points=self.team_players_0_xy)
        self.pitch_team1_players_xy = self.transformer.transform_points(points=self.team_players_1_xy)
