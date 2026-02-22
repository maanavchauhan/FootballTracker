"""
Configuration and constants for the Football AI analysis system.

Centralizes all global settings, detection class IDs, model identifiers,
and processing parameters used throughout the pipeline.
"""

import os

# ── Runtime Environment ─────────────────────────────────────────────────────
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

# ── Video Paths ─────────────────────────────────────────────────────────────
SOURCE_VIDEO_PATH = "/kaggle/working/121364_0.mp4"
TARGET_VIDEO_PATH = "/kaggle/working/121364_0_result.mp4"

# ── Detection Class IDs ─────────────────────────────────────────────────────
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# ── Processing Parameters ───────────────────────────────────────────────────
STRIDE = 30                          # Sample every STRIDE-th frame for crops
PLAYER_IN_POSSESSION_PROXIMITY = 40  # Max distance (px) to count as possession
FPS = 25                             # Frames per second of the source video

# ── Model Identifiers ──────────────────────────────────────────────────────
DETECTION_MODEL_ID = "football-detection-ysgxf/3"
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi-jufj9/1"
SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"

# ── Embedding / Classification ──────────────────────────────────────────────
BATCH_SIZE = 32

# ── Pitch Dimension Scaling ─────────────────────────────────────────────────
# Maps raw pitch coordinates to real-world metres
X_SCALE = 105 / 12000
Y_SCALE = 68 / 7000
