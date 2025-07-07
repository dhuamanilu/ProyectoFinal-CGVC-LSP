# words_recognition/scripts/test_word_interface.py

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
from words_recognition.scripts.training_word_mode import TrainingModeWordsWindow

# -------- Configuración de rutas (sin cambios) --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'word_model.h5')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'word_label_encoder.pkl')

EXPECTED_FEATURES = (33+21+21) * 3

# --- CAMBIO: La función ahora devuelve los recursos cargados ---
def load_resources():
    try:
        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return model, encoder, holistic
    except Exception as e:
        messagebox.showerror("Error de Carga", f"No se pudieron cargar los recursos del modelo: {e}")
        return None, None, None

# --- Lógica de predicción (sin cambios) ---
def extract_landmarks(frames, holistic_instance):
    seq = []
    for frame in frames:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic_instance.process(img_rgb)
        features = []
        for lm_list, count in [
            (res.pose_landmarks.landmark if res.pose_landmarks else [], 33),
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
        
        self.is_preview_active = False
        self.is_recording = False
        self.frames = []
        
        # --- CAMBIO: Cargar y almacenar todos los recursos en la instancia principal ---
        self.model, self.encoder, self.holistic = load_resources()
        self.cap = cv2.VideoCapture(0)

        # UI (sin cambios)
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        ttk.Label(controls_frame, text="Duración (s):").pack(side=tk.LEFT, padx=(0, 5))
        self.duration = tk.IntVar(value=5)
        ttk.Spinbox(controls_frame, from_=1, to=20, textvariable=self.duration, width=5).pack(side=tk.LEFT)
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        self.start_btn = ttk.Button(buttons_frame, text="Iniciar Predicción", command=self.start_capture)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.stop_btn = ttk.Button(buttons_frame, text="Detener", command=self.stop_capture, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.train_btn = ttk.Button(buttons_frame, text="Modo Entrenamiento", command=self.open_training_mode)
        self.train_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.video_label = ttk.Label(main_frame, text="Cámara desactivada", font=("Arial", 12))
        self.video_label.pack(pady=10)
        self.result_label = ttk.Label(main_frame, text="Resultado: --", font=('Arial', 14), anchor="center")
        self.result_label.pack(fill=tk.X, pady=10)

        if not self.model or not self.cap.isOpened():
             self.start_btn.config(state="disabled")
             self.train_btn.config(state="disabled")

        self.resume_preview()
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def open_training_mode(self):
        self.is_preview_active = False # Pausa el dibujado de la cámara en esta ventana
        self.root.withdraw()
        # --- CAMBIO: Pasar todos los recursos a la ventana de entrenamiento ---
        TrainingModeWordsWindow(self, self.model, self.encoder, self.holistic, self.cap)

    # --- CAMBIO: renombrado para claridad, reanuda el dibujado ---
    def resume_preview(self):
        self.is_preview_active = True
        self.update_video_preview()

    def update_video_preview(self):
        # El bucle ahora solo se dibuja si la ventana está activa
        if not self.is_preview_active or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((480, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")
            if self.is_recording:
                self.frames.append(frame.copy())
        
        self.root.after(20, self.update_video_preview)

    def predict(self):
        # --- CAMBIO: Pasa la instancia de holistic a la función ---
        seq = extract_landmarks(self.frames, self.holistic)
        if not seq:
            messagebox.showerror("Error", "No se pudieron extraer landmarks.")
            return
        max_len = self.model.input_shape[1]
        X = pad_sequence(seq, max_len)
        X = np.expand_dims(X, axis=0)
        pred = self.model.predict(X)
        idx = np.argmax(pred, axis=1)[0]
        palabra = self.encoder.inverse_transform([idx])[0]
        prob = pred[0, idx]
        self.result_label.config(text=f"Resultado: {palabra} ({prob:.2f})")

    def on_closing(self):
        self.is_preview_active = False
        if self.cap: self.cap.release()
        if self.holistic: self.holistic.close()
        self.root.destroy()
    
    # Lógica de start/stop (sin cambios relevantes)
    def start_capture(self):
        if not self.model: messagebox.showerror("Error", "El modelo no está cargado."); return
        if not self.is_preview_active: messagebox.showwarning("Aviso", "La cámara no está activa."); return
        self.is_recording = True; self.frames = []; self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal'); self.train_btn.config(state='disabled')
        self.root.after(self.duration.get() * 1000, self.stop_capture)

    def stop_capture(self):
        if not self.is_recording: return
        self.is_recording = False; self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled'); self.train_btn.config(state='normal')
        if not self.frames: messagebox.showwarning("Aviso", "No se capturaron frames."); return
        self.predict()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()