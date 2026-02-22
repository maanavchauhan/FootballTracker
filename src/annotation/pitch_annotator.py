"""
Pitch-level annotation.

Functions for rendering the 2-D pitch view with player/ball positions,
Voronoi diagrams, offside lines, and heatmap overlays.
"""

import cv2
import numpy as np
import supervision as sv  # type: ignore

from typing import Optional

from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_paths_on_pitch,
    draw_pitch_voronoi_diagram,
)

from entity.frame_detections import FrameDetections
from entity.pitch_detections import PitchDetections

CONFIG = SoccerPitchConfiguration()


# ─── Pitch homography view ──────────────────────────────────────────────────

def homography_pitch(
    frame,
    f: FrameDetections,
    p: PitchDetections,
    offside=False,
    last_defender_positions=None,
    potential_offsides=None,
):
    """
    Draws a top-down pitch with projected player/ball/referee positions.
    Optionally overlays offside lines.
    """
    if potential_offsides is None:
        potential_offsides = []

    annotated_frame = draw_pitch(CONFIG)

    # Ball
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_frame
    )

    # Team 0
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_all_players_xy[f.all_players_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame
    )

    # Team 1
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_all_players_xy[f.all_players_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame
    )

    # Referees
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_referees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame
    )

    if offside:
        if last_defender_positions is None:
            from analysis.offside import find_last_defender
            last_defender_positions, potential_offsides = find_last_defender(f, p)

        # Vertical offside lines
        x_coord = last_defender_positions[0]
        vertical_line_path = np.array([
            [x_coord, 0],
            [x_coord, CONFIG.width]
        ])
        annotated_frame = draw_paths_on_pitch(config=CONFIG, paths=[vertical_line_path], pitch=annotated_frame)

        x_coord = last_defender_positions[1]
        vertical_line_path = np.array([
            [x_coord, 0],
            [x_coord, CONFIG.width]
        ])
        annotated_frame = draw_paths_on_pitch(config=CONFIG, paths=[vertical_line_path], pitch=annotated_frame)

    return annotated_frame


# ─── Custom Voronoi diagram with smooth blending ────────────────────────────

def draw_pitch_voronoi_diagram_2(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders a smooth (tanh-blended) Voronoi territory diagram on the pitch.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coords, y_coords):
        return np.sqrt(
            (xy[:, 0][:, None, None] * scale - x_coords) ** 2 +
            (xy[:, 1][:, None, None] * scale - y_coords) ** 2
        )

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    steepness = 15  # Sharper transition
    distance_ratio = min_distances_team_2 / np.clip(
        min_distances_team_1 + min_distances_team_2, a_min=1e-5, a_max=None
    )
    blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

    for c in range(3):
        voronoi[:, :, c] = (
            blend_factor * team_1_color_bgr[c] + (1 - blend_factor) * team_2_color_bgr[c]
        ).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)
    return overlay


# ─── Heatmap overlay ────────────────────────────────────────────────────────

def draw_pitch_heatmap_on_frame(overlay, transformer_inverse, frame, opacity=0.6):
    """
    Warps a pitch-space overlay onto the original camera frame and
    blends them together.
    """
    resized_overlay = cv2.resize(overlay, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
    warped_overlay = transformer_inverse.transform_image(
        resized_overlay, resolution_wh=(frame.shape[1], frame.shape[0])
    )
    final_overlay = cv2.addWeighted(warped_overlay, opacity, frame, 1 - opacity, 0)
    return final_overlay


# ─── Colour utility ─────────────────────────────────────────────────────────

def lighten_color(hex_color, factor=0.3):
    """
    Returns a lighter variant of a hex colour string.
    """
    rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    lighter_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)


# ─── Full Voronoi computation helper ────────────────────────────────────────

def compute_voronoi(f: FrameDetections, p: PitchDetections):
    """
    Builds a complete Voronoi diagram pitch image with player and ball
    markers for a single frame.
    """
    annotated_frame = draw_pitch(
        config=CONFIG,
        background_color=sv.Color.WHITE,
        line_color=sv.Color.BLACK
    )

    annotated_frame = draw_pitch_voronoi_diagram_2(
        config=CONFIG,
        team_1_xy=p.pitch_players_xy[f.players_detections.class_id == 0],
        team_2_xy=p.pitch_players_xy[f.players_detections.class_id == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=annotated_frame
    )

    # Ball marker
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.WHITE,
        radius=8,
        thickness=1,
        pitch=annotated_frame
    )

    # Team 0 players
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_players_xy[f.players_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame
    )

    # Team 1 players
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=p.pitch_players_xy[f.players_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame
    )

    return annotated_frame
