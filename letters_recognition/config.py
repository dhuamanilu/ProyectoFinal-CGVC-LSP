# letters_recognition/config.py

from pathlib import Path

BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / "dataset"
MODEL_PATH     = BASE_DIR / "models" / "modelo_letras.pkl"
ENCODER_PATH   = BASE_DIR / "models" / "labels_encoder.pkl"

# MediaPipe settings
MP_SETTINGS = {
    "static_image_mode": False,
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}
