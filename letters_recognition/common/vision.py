# lsp_app/common/vision.py
"""
Carga el modelo y MediaPipe Hands una sola vez y los expone como
variables globales que los módulos de interfaz pueden reutilizar.
"""
import threading
import joblib
import mediapipe as mp

from .assets import MODEL_PATH, ENCODER_PATH

model = label_encoder = mp_hands = mp_drawing = hands = None


def _load_sync():
    """Carga sincrónicamente modelo y MediaPipe Hands."""
    global model, label_encoder, mp_hands, mp_drawing, hands

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def async_init_vision(on_ready):
    """
    Arranca _load_sync() en un thread para no congelar la UI.
    Llama a `on_ready()` (callback) cuando termina.
    """
    def _target():
        _load_sync()
        on_ready()

    threading.Thread(target=_target, daemon=True).start()
