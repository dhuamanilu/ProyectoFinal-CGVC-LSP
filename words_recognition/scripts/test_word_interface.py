
"""
Interfaz gráfica con Tkinter y vista previa de webcam para probar el modelo de reconocimiento de palabras en tiempo real.
Selecciona duración, presiona Iniciar/Detener. Muestra video en vivo y resultado.
"""
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import mediapipe as mp
from PIL import Image, ImageTk

# -------- Configuración de rutas --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'word_model.h5')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'word_label_encoder.pkl')

# -------- MediaPipe Holistic --------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
EXPECTED_FEATURES = (33+21+21) * 3

# Carga modelo y encoder global

def load_resources():
    global model, encoder
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# Extrae landmarks de frames

def extract_landmarks(frames):
    seq = []
    for frame in frames:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(img_rgb)
        features = []
        for lm_list, count in [
            (res.pose_landmarks.landmark if res.pose_landmarks else [], 33),
            #(res.face_landmarks.landmark if res.face_landmarks else [], 468),
            (res.left_hand_landmarks.landmark if res.left_hand_landmarks else [], 21),
            (res.right_hand_landmarks.landmark if res.right_hand_landmarks else [], 21)
        ]:
            if lm_list:
                for lm in lm_list:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * count * 3)
        if len(features) == EXPECTED_FEATURES:
            seq.append(features)
    return seq

# Padding/truncado

def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return np.array(seq[:max_len], dtype=np.float32)
    pad = np.zeros((max_len - len(seq), EXPECTED_FEATURES), dtype=np.float32)
    return np.vstack([seq, pad])

class App:
    def __init__(self, root):
        self.root = root
        root.title("Reconocimiento de Palabras")
        root.resizable(False, False)

        # Duración
        ttk.Label(root, text="Duración (s):").grid(row=0, column=0, sticky='w', padx=5)
        self.duration = tk.IntVar(value=5)
        ttk.Spinbox(root, from_=1, to=20, textvariable=self.duration, width=5).grid(row=0, column=1)

        # Botones
        self.start_btn = ttk.Button(root, text="Iniciar", command=self.start_capture)
        self.start_btn.grid(row=1, column=0, padx=5, pady=5)
        self.stop_btn = ttk.Button(root, text="Detener", command=self.stop_capture, state='disabled')
        self.stop_btn.grid(row=1, column=1)

        # Video Label
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=2, column=0, columnspan=2)

        # Resultado
        self.result_label = ttk.Label(root, text="Resultado: --", font=('Arial', 14))
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

        self.recording = False
        self.frames = []
        self.cap = cv2.VideoCapture(0)
        load_resources()
        self.update_video_preview()

    def update_video_preview(self):
        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((400, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            if self.recording:
                self.frames.append(frame.copy())
        self.root.after(30, self.update_video_preview)

    def start_capture(self):
        self.recording = True
        self.frames = []
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        # Auto stop tras duración
        self.root.after(self.duration.get() * 1000, self.stop_capture)

    def stop_capture(self):
        if not self.recording:
            return
        self.recording = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        if not self.frames:
            messagebox.showwarning("Aviso", "No se capturaron frames.")
            return
        self.predict()

    def predict(self):
        seq = extract_landmarks(self.frames)
        if not seq:
            messagebox.showerror("Error", "No se pudieron extraer landmarks.")
            return
        max_len = model.input_shape[1]
        X = pad_sequence(seq, max_len)
        X = np.expand_dims(X, axis=0)
        pred = model.predict(X)
        idx = np.argmax(pred, axis=1)[0]
        palabra = encoder.inverse_transform([idx])[0]
        prob = pred[0, idx]
        self.result_label.config(text=f"Resultado: {palabra} ({prob:.2f})")

    def __del__(self):
        self.cap.release()
        holistic.close()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

