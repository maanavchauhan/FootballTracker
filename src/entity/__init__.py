"""
Entity package — detection types used in football frame analysis.

Hierarchy:
    Detections (base)
    ├── Ball
    ├── Players
    └── Goalkeepers

    FrameDetections   — orchestrates per-frame detection processing
    PitchDetections   — handles pitch-view coordinate transformations
"""

from entity.detections import Detections
from entity.ball import Ball
from entity.players import Players
from entity.goalkeepers import Goalkeepers
from entity.frame_detections import FrameDetections
from entity.pitch_detections import PitchDetections

__all__ = [
    "Detections",
    "Ball",
    "Players",
    "Goalkeepers",
    "FrameDetections",
    "PitchDetections",
]
