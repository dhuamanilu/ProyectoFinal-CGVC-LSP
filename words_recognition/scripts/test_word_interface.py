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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'word_model.h5')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'word_label_encoder.pkl')

EXPECTED_FEATURES = (33 + 21 + 21) * 3

PALETTE = {
    "header": "#ff6b6b",
    "primary": "#4285f4",
    "danger": "#ea4335",
    "bg": "#ffffff"
}

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
    CAM_WIDTH, CAM_HEIGHT = 480, 360

    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Palabras")
        self.root.configure(bg=PALETTE["bg"])
        self.root.geometry("850x520")
        self.is_recording = False
        self.is_preview_active = True
        self.frames = []

        self.model, self.encoder, self.holistic = load_resources()
        self.cap = cv2.VideoCapture(0)

        self.duration = tk.IntVar(value=5)

        self._init_styles()
        self._build_ui()

        if not self.model or not self.cap.isOpened():
            self.start_btn.config(state="disabled")
            self.train_btn.config(state="disabled")

        self.update_video_preview()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Header.TLabel", background=PALETTE["header"], foreground="white", font=("Arial", 16, "bold"), anchor="center")
        style.configure("Accent.TButton", background=PALETTE["primary"], foreground="white", font=("Arial", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#2a68c7")])

    def _build_ui(self):
        ttk.Label(self.root, text="Reconocimiento de Palabras en Lengua de Señas Peruana", style="Header.TLabel").pack(fill="x")

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Cámara (Izquierda)
        cam_frame = ttk.Frame(main_frame)
        cam_frame.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_cam = tk.Canvas(cam_frame, width=self.CAM_WIDTH, height=self.CAM_HEIGHT, bg="#e9ecef")
        self.canvas_cam.pack()
        self.lbl_countdown = tk.Label(self.canvas_cam, text="", font=("Arial", 80, "bold"), fg="white", bg="black")

        # Controles (Derecha)
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        ttk.Label(ctrl_frame, text="Duración (s):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Spinbox(ctrl_frame, from_=1, to=20, textvariable=self.duration, width=5).pack(anchor="w", pady=(0, 20))
        self.start_btn = ttk.Button(ctrl_frame, text="Iniciar Predicción", command=self.start_capture, style="Accent.TButton")
        self.start_btn.pack(fill="x", pady=5, ipady=6)
        self.stop_btn = ttk.Button(ctrl_frame, text="Detener", command=self.stop_capture, state='disabled')
        self.stop_btn.pack(fill="x", pady=5, ipady=6)
        self.train_btn = ttk.Button(ctrl_frame, text="Modo Entrenamiento", command=self.open_training_mode)
        self.train_btn.pack(fill="x", pady=5, ipady=6)

        # Resultado
        self.result_label = ttk.Label(self.root, text="Resultado: --", font=("Arial", 14, "bold"), anchor="center", relief="groove")
        self.result_label.pack(fill="x", padx=10, pady=(5, 10))

    def open_training_mode(self):
        self.is_preview_active = False
        self.root.withdraw()
        TrainingModeWordsWindow(self, self.model, self.encoder, self.holistic, self.cap)

    def resume_preview(self):
        self.is_preview_active = True
        self.update_video_preview()

    def update_video_preview(self):
        if not self.is_preview_active or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((self.CAM_WIDTH, self.CAM_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas_cam.imgtk = imgtk
            self.canvas_cam.create_image(0, 0, anchor=tk.NW, image=imgtk)
            if self.is_recording:
                self.frames.append(frame.copy())
        self.root.after(20, self.update_video_preview)

    def show_countdown_and_record(self):
        for i in range(3, 0, -1):
            self.lbl_countdown.place(x=self.CAM_WIDTH//2, y=self.CAM_HEIGHT//2, anchor="center")
            self.lbl_countdown.config(text=str(i))
            time.sleep(1)
        self.lbl_countdown.config(text="¡YA!")
        time.sleep(1)
        self.lbl_countdown.place_forget()

        self.is_recording = True
        self.frames = []
        time.sleep(self.duration.get())
        self.is_recording = False

        self.root.after(0, self.predict)

    def start_capture(self):
        if not self.model:
            messagebox.showerror("Error", "El modelo no está cargado.")
            return
        if not self.is_preview_active:
            messagebox.showwarning("Aviso", "La cámara no está activa.")
            return
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.train_btn.config(state="disabled")

        threading.Thread(target=self.show_countdown_and_record, daemon=True).start()

    def stop_capture(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.train_btn.config(state="normal")
        if not self.frames:
            messagebox.showwarning("Aviso", "No se capturaron frames.")
            return
        self.predict()

    def predict(self):
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
        if self.cap:
            self.cap.release()
        if self.holistic:
            self.holistic.close()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()