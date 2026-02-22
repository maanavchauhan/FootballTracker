"""
Annotation package — frame and pitch visualisation helpers.
"""

from annotation.annotators import get_annotators
from annotation.frame_annotator import frame_annotation
from annotation.pitch_annotator import (
    homography_pitch,
    draw_pitch_voronoi_diagram_2,
    draw_pitch_heatmap_on_frame,
    compute_voronoi,
    lighten_color,
)

__all__ = [
    "get_annotators",
    "frame_annotation",
    "homography_pitch",
    "draw_pitch_voronoi_diagram_2",
    "draw_pitch_heatmap_on_frame",
    "compute_voronoi",
    "lighten_color",
]
