"""
Model loading.

Provides functions to load the three ML models used by the pipeline:
    1. Object detection (Roboflow YOLO)
    2. Field key-point detection (Roboflow YOLO)
    3. Visual embeddings (SigLIP) for team classification
"""

import os
import torch

from inference import get_model
from transformers import AutoProcessor, SiglipVisionModel

from config import (
    DETECTION_MODEL_ID,
    FIELD_DETECTION_MODEL_ID,
    SIGLIP_MODEL_PATH,
)


def get_api_key():
    """
    Retrieves the Roboflow API key.

    Tries the ROBOFLOW_API_KEY environment variable first; falls back
    to Kaggle UserSecretsClient when running on Kaggle.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key:
        return api_key

    # Fallback: Kaggle secrets (original notebook approach)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("Roboflow")
    except Exception:
        raise RuntimeError(
            "Roboflow API key not found. Set the ROBOFLOW_API_KEY "
            "environment variable or configure Kaggle secrets."
        )


def load_detection_model(api_key=None):
    """
    Loads the object-detection model (players, ball, goalkeepers, referees).
    """
    if api_key is None:
        api_key = get_api_key()
    # Load Object Detection Model
    return get_model(model_id=DETECTION_MODEL_ID, api_key=api_key)


def load_field_detection_model(api_key=None):
    """
    Loads the field key-point detection model (pitch lines / corners).
    """
    if api_key is None:
        api_key = get_api_key()
    # Load Field (Keypoint) Detection Model
    return get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=api_key)


def load_embeddings_model():
    """
    Loads the SigLIP vision model and processor for player-crop embeddings.

    Returns:
        (model, processor, device): The loaded model, its processor, and
        the device string ('cuda' or 'cpu').
    """
    # Load Classification Model (SigLIP)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    return model, processor, device
