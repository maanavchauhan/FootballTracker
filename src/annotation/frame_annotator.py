"""
Frame annotation.

Draws detection overlays (ellipses, labels, possession triangles,
offside indicators) onto a single video frame.
"""

import numpy as np
import supervision as sv  # type: ignore

from config import PLAYER_IN_POSSESSION_PROXIMITY
from entity.frame_detections import FrameDetections
from entity.pitch_detections import PitchDetections
from analysis.possession import get_player_in_possession
from analysis.offside import find_last_defender
from annotation.annotators import get_annotators

from sports.configs.soccer import SoccerPitchConfiguration

# Module-level annotators (created once, reused every call)
_annotators = get_annotators()
CONFIG = SoccerPitchConfiguration()


def frame_annotation(
    frame,
    f: FrameDetections,
    previous_possesion=None,
    offside=False,
    last_defender_positions=None,
    potential_offsides=None,
    cumulative_distances=None,
    instantaneous_speed=None,
    p: PitchDetections = None,
):
    """
    Produces an annotated copy of *frame* with detection ellipses,
    labels, ball triangle, possession indicator, and (optionally)
    offside lines.

    Returns:
        annotated_frame: The annotated image.
        previous_possesion: Updated possession detection for the next call.
    """
    if potential_offsides is None:
        potential_offsides = []

    ann = _annotators
    POSSESION_ID = get_player_in_possession(f, proximity=PLAYER_IN_POSSESSION_PROXIMITY)

    if offside:
        # Compute offside data if not provided
        if last_defender_positions is None:
            last_defender_positions, potential_offsides = find_last_defender(f, p)

        offside_mask = np.isin(f.all_detections.tracker_id, potential_offsides)
        offside_detections = f.all_detections[offside_mask]
        non_offside_detections = f.all_detections[~offside_mask]

        # Build labels (with optional distance / speed info)
        if cumulative_distances is not None:
            offside_labels = [
                f"#{tid} | {cumulative_distances.get(tid, 0):.2f}m | {instantaneous_speed.get(tid, 0):.2f}km/h"
                for tid in offside_detections.tracker_id
            ]
            non_offside_labels = [
                f"#{tid} | {cumulative_distances.get(tid, 0):.2f}m | {instantaneous_speed.get(tid, 0):.2f}km/h"
                for tid in non_offside_detections.tracker_id
            ]
        else:
            offside_labels = [f"#{tid}" for tid in offside_detections.tracker_id]
            non_offside_labels = [f"#{tid}" for tid in non_offside_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ann["ellipse"].annotate(scene=annotated_frame, detections=non_offside_detections)
        annotated_frame = ann["offside_ellipse"].annotate(scene=annotated_frame, detections=offside_detections)
        annotated_frame = ann["triangle"].annotate(scene=annotated_frame, detections=f.ball_detections)
        annotated_frame = ann["label"].annotate(scene=annotated_frame, detections=non_offside_detections, labels=non_offside_labels)
        annotated_frame = ann["offside_label"].annotate(scene=annotated_frame, detections=offside_detections, labels=offside_labels)

        # Draw offside reference lines on the frame
        offside_points = np.array([
            [last_defender_positions[0], 0],
            [last_defender_positions[0], CONFIG.width],
            [last_defender_positions[1], 0],
            [last_defender_positions[1], CONFIG.width]
        ])

        frame_coordinates = p.transformer_inverse.transform_points(points=offside_points)
        key_points = sv.KeyPoints(xy=frame_coordinates[np.newaxis, ...])

        edges = [(0, 3), (1, 2)]
        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            thickness=2,
            edges=edges
        )

        annotated_frame = ann["vertex"].annotate(scene=annotated_frame, key_points=key_points)
        annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=key_points)
    else:
        # Simple annotation (no offside overlays)
        annotated_frame = frame.copy()
        annotated_frame = ann["ellipse"].annotate(scene=annotated_frame, detections=f.all_detections)
        annotated_frame = ann["triangle"].annotate(scene=annotated_frame, detections=f.ball_detections)
        annotated_frame = ann["label"].annotate(scene=annotated_frame, detections=f.all_detections, labels=f.labels)

    # ── Possession indicator ────────────────────────────────────────────
    if POSSESION_ID != -1:
        if len(f.ball_detections) == 0 and previous_possesion is not None:
            possesion_detection = previous_possesion
        else:
            possesion_detection = f.all_detections[f.all_detections.tracker_id == POSSESION_ID]
            possesion_detection.xyxy = sv.pad_boxes(xyxy=possesion_detection.xyxy, px=10)
            previous_possesion = possesion_detection
        annotated_frame = ann["player_triangle"].annotate(scene=annotated_frame, detections=possesion_detection)

    return annotated_frame, previous_possesion
