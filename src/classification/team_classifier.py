"""
Team classification pipeline.

Extracts player crops from video frames, computes visual embeddings
(SigLIP), reduces dimensionality (UMAP), clusters into two teams
(KMeans), and fits the TeamClassifier from the sports library.
"""

import umap
import torch
import numpy as np
import supervision as sv  # type: ignore

from tqdm import tqdm
from more_itertools import chunked
from sklearn.cluster import KMeans

from sports.common.team import TeamClassifier

from config import STRIDE, PLAYER_ID, BATCH_SIZE


# ─── Crop extraction ────────────────────────────────────────────────────────

def extract_crops(source_video_path: str, detection_model):
    """
    Samples every STRIDE-th frame and extracts bounding-box crops for
    all detected players.
    """
    # Stride to sample every STRIDE-th frame
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []

    for frame_sample in tqdm(frame_generator, desc='Collecting crops'):
        result = detection_model.infer(frame_sample, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame_sample, xyxy) for xyxy in detections.xyxy]

    return crops


# ─── Team classifier fitting ────────────────────────────────────────────────

def fit_team_classifier(source_video_path: str, detection_model, device="cuda"):
    """
    Collects player crops from the video and fits a TeamClassifier.

    Returns:
        team_classifier: A fitted TeamClassifier instance.
    """
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []

    # Collect crops for team classification from video frames
    for frame in tqdm(frame_generator, desc='Collecting crops for team classification'):
        result = detection_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    return team_classifier
