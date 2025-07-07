# words_recognition/scripts/training_word_mode.py

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import os
import time

# --- CAMBIO: Eliminada la carga de modelos y la inicialización de MediaPipe de aquí ---

# --- Configuración de rutas (sin cambios) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

EXPECTED_FEATURES = (33 + 21 + 21) * 3
SELECTED_WORDS = [
    "HOLA", "BIEN", "AMIGO", "CASA", "DINERO",
    "HACER", "IR", "LO SIENTO", "MUCHO", "UNO"
]

# --- Lógica de predicción (ahora recibe la instancia de holistic) ---
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

class TrainingModeWordsWindow:
    CAM_WIDTH, CAM_HEIGHT = 420, 320
    EXAMPLE_WIDTH, EXAMPLE_HEIGHT = 220, 170
    PALETTE = {"primary": "#4285f4", "danger": "#ea4335", "success": "#28a745", 
               "warning": "#ff9f0a", "gray": "#6c757d", "bg": "#ffffff", "header": "#ff6b6b"}

    # --- CAMBIO: El constructor ahora recibe los recursos compartidos ---
    def __init__(self, parent, model, encoder, holistic, cap):
        self.parent = parent
        self.model = model
        self.encoder = encoder
        self.holistic = holistic
        self.cap = cap

        self.target_word = None
        self.is_recording = False
        self.is_preview_active = True # Activa su propia vista previa
        self.is_playing_example = False
        self.recorded_frames = []
        self.example_videos = []
        self.example_idx = 0
        
        self.top = tk.Toplevel(parent.root)
        self.top.title("Modo Entrenamiento – Palabras (LSP)")
        self.top.geometry("1050x600")
        self.top.resizable(False, False)
        self.top.configure(background=self.PALETTE["bg"])
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        
        self.duration = tk.IntVar(value=5)

        self._init_styles()
        self._build_ui()
        # --- CAMBIO: Ya no crea una nueva captura, usa la existente ---
        threading.Thread(target=self._update_webcam_feed, daemon=True).start()

    def close(self):
        self.is_preview_active = False
        self.is_playing_example = False 
        self.top.destroy()
        self.parent.root.deiconify()
        # --- CAMBIO: Llama a la función del padre para reanudar su vista ---
        self.parent.resume_preview()
    
    # El resto del código permanece casi idéntico, solo cambia cómo se llama a extract_landmarks
    # ...
    def predict_and_show_result(self):
        if not self.recorded_frames:
            self.lbl_result.config(text="Error: No se grabó nada.", background=self.PALETTE["danger"], foreground="white")
            self.status_label.config(text="Práctica fallida. Inténtalo de nuevo.")
            return
        
        # --- CAMBIO: Pasa la instancia de holistic a la función de extracción ---
        seq = extract_landmarks(self.recorded_frames, self.holistic)
        
        if not seq:
            self.lbl_result.config(text="Error: No se detectó ninguna mano.", background=self.PALETTE["danger"], foreground="white")
            self.status_label.config(text="Asegúrate de que tu mano sea visible para la cámara.")
            return

        X = pad_sequence(seq, self.model.input_shape[1])
        X = np.expand_dims(X, axis=0)
        pred = self.model.predict(X)
        idx = np.argmax(pred, axis=1)[0]
        palabra_predicha = self.encoder.inverse_transform([idx])[0]
        prob = pred[0, idx]

        if palabra_predicha == self.target_word and prob >= 0.50:
            result_text = f"¡Excelente! {palabra_predicha} ({prob:.0%})"
            bg_color = self.PALETTE["success"]
            self.status_label.config(text="¡Práctica completada con éxito!")
        else:
            result_text = f"Detectado: {palabra_predicha} ({prob:.0%}). Intenta de nuevo."
            bg_color = self.PALETTE["danger"]
            self.status_label.config(text="El resultado no coincide. ¡Sigue practicando!")
        self.lbl_result.config(text=result_text, background=bg_color, foreground="white")

    # --- El resto de los métodos (UI, hilos, etc.) no necesitan cambios lógicos ---
    # Se incluyen aquí para que el archivo esté completo.

    def _init_styles(self):
        style = ttk.Style(self.top); style.theme_use("clam")
        style.configure("TFrame", background=self.PALETTE["bg"])
        style.configure("TLabel", background=self.PALETTE["bg"], font=("Arial", 10))
        style.configure("Header.TLabel", background=self.PALETTE["header"], foreground="white", font=("Arial", 16, "bold"), anchor="center")
        style.configure("TButton", font=("Arial", 10, "bold"))
        style.configure("Accent.TButton", background=self.PALETTE["primary"], foreground="white", font=("Arial", 10, "bold"))
        style.map("Accent.TButton", background=[("active", self.PALETTE["success"])])
        style.configure("Pic.TLabel", relief="solid", borderwidth=1, anchor="center")
        style.configure("TLabelframe", background=self.PALETTE["bg"])
        style.configure("TLabelframe.Label", background=self.PALETTE["bg"], font=("Arial", 11, "bold"))

    def _build_ui(self):
        header = ttk.Frame(self.top, height=50); header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        ttk.Label(header, text="Modo Entrenamiento – Palabras en Lengua de Señas Peruana", style="Header.TLabel").pack(fill="both", expand=True)
        cam_frame = ttk.Frame(self.top, width=self.CAM_WIDTH, height=self.CAM_HEIGHT); cam_frame.grid(row=1, column=0, padx=20, pady=10, sticky="n"); cam_frame.grid_propagate(False)
        self.canvas_webcam = tk.Canvas(cam_frame, width=self.CAM_WIDTH, height=self.CAM_HEIGHT, bg="#e9ecef", bd=0, highlightthickness=0); self.canvas_webcam.pack()
        self.lbl_countdown = tk.Label(self.canvas_webcam, text="", font=("Arial", 80, "bold"), fg="white", bg="black")
        feedback_frame = ttk.Frame(self.top); feedback_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")
        ttk.Label(feedback_frame, text="Palabra Objetivo", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        self.lbl_objetivo = ttk.Label(feedback_frame, text="-", width=12, font=("Arial", 40, "bold"), style="Pic.TLabel", anchor="center"); self.lbl_objetivo.pack(pady=(0, 20), ipady=10)
        example_gallery = ttk.LabelFrame(feedback_frame, text="Video de Ejemplo"); example_gallery.pack(fill="x", padx=8)
        self.canvas_example = tk.Canvas(example_gallery, width=self.EXAMPLE_WIDTH, height=self.EXAMPLE_HEIGHT, bg="#e9ecef", bd=0, highlightthickness=0); self.canvas_example.pack(padx=5, pady=5)
        self.btn_play_example = ttk.Button(example_gallery, text="▶ Reproducir Ejemplo", command=self.play_example_video, state="disabled"); self.btn_play_example.pack(pady=(0, 5))
        controls_frame = ttk.Frame(self.top); controls_frame.grid(row=1, column=2, padx=20, pady=10, sticky="n")
        ttk.Label(controls_frame, text="1. Elige una palabra:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
        self.combo_word = ttk.Combobox(controls_frame, state="readonly", values=SELECTED_WORDS, font=("Arial", 11), width=20); self.combo_word.pack(anchor="w", pady=(0, 20)); self.combo_word.bind("<<ComboboxSelected>>", self._on_word_select)
        ttk.Label(controls_frame, text="2. Ajusta la duración (s):", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Spinbox(controls_frame, from_=2, to=15, textvariable=self.duration, width=5, font=("Arial", 11)).pack(anchor="w", pady=(0, 20))
        self.btn_practice = ttk.Button(controls_frame, text="¡Iniciar Práctica!", command=self.start_practice, style="Accent.TButton", state="disabled"); self.btn_practice.pack(fill="x", ipady=8, pady=(10, 5))
        ttk.Button(controls_frame, text="⬅️ Volver", command=self.close).pack(fill="x", ipady=5, pady=5)
        self.lbl_result = ttk.Label(self.top, text="Resultado: --", font=("Arial", 14, "bold"), anchor="center", padding=10, relief="groove"); self.lbl_result.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=10)
        self.status_label = ttk.Label(self.top, text="Estado: Selecciona una palabra para comenzar.", anchor="w", relief="sunken"); self.status_label.grid(row=3, column=0, columnspan=3, sticky="ew")

    def _update_webcam_feed(self):
        while self.is_preview_active:
            if not self.cap or not self.cap.isOpened(): continue
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            # Solo dibuja landmarks si la ventana está activa, para no consumir recursos innecesariamente
            if self.is_preview_active:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                if results.pose_landmarks: import mediapipe as mp; mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks: import mediapipe as mp; mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks: import mediapipe as mp; mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            if self.is_recording: self.recorded_frames.append(frame.copy())
            self._update_canvas(self.canvas_webcam, frame, self.CAM_WIDTH, self.CAM_HEIGHT)
    
    def _on_word_select(self, event=None):
        self.target_word = self.combo_word.get(); self.lbl_objetivo.config(text=self.target_word); self.btn_practice.config(state="normal")
        self.lbl_result.config(text="Resultado: --", background=self.top.cget('bg'), foreground="black")
        self.status_label.config(text=f"Palabra objetivo '{self.target_word}' seleccionada. ¡Listo para practicar!")
        word_dir = os.path.join(RAW_DATA_DIR, self.target_word)
        self.example_videos = []
        if os.path.isdir(word_dir):
            for f in sorted(os.listdir(word_dir)):
                if f.endswith('.mp4') and "ORACION" not in f.upper(): self.example_videos.append(os.path.join(word_dir, f))
        if self.example_videos:
            self.btn_play_example.config(state="normal"); self._show_first_frame_of_example()
        else:
            self.btn_play_example.config(state="disabled"); self.canvas_example.delete("all"); self.canvas_example.create_text(self.EXAMPLE_WIDTH/2, self.EXAMPLE_HEIGHT/2, text="No hay videos", fill="gray")

    def _show_first_frame_of_example(self):
        cap = cv2.VideoCapture(self.example_videos[self.example_idx]); ret, frame = cap.read()
        if ret: self._update_canvas(self.canvas_example, frame, self.EXAMPLE_WIDTH, self.EXAMPLE_HEIGHT)
        cap.release()

    def play_example_video(self):
        if self.is_playing_example or not self.example_videos: return
        threading.Thread(target=self._play_example_thread, daemon=True).start()

    def _play_example_thread(self):
        self.is_playing_example = True; self.btn_play_example.config(state="disabled"); self.status_label.config(text="Reproduciendo video de ejemplo...")
        cap = cv2.VideoCapture(self.example_videos[self.example_idx])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or not self.is_playing_example: break
            self._update_canvas(self.canvas_example, frame, self.EXAMPLE_WIDTH, self.EXAMPLE_HEIGHT)
            time.sleep(1/30)
        cap.release(); self.is_playing_example = False; self.btn_play_example.config(state="normal"); self.status_label.config(text="Video de ejemplo finalizado.")

    def start_practice(self):
        if not self.target_word: return
        threading.Thread(target=self._practice_thread, daemon=True).start()

    def _practice_thread(self):
        self.lbl_result.config(text="Prepárate...", background=self.top.cget('bg'), foreground="black"); self.btn_practice.config(state="disabled"); self.combo_word.config(state="disabled"); self.status_label.config(text="La práctica comenzará en...")
        for i in range(3, 0, -1):
            self.lbl_countdown.place(x=self.CAM_WIDTH/2, y=self.CAM_HEIGHT/2, anchor="center"); self.lbl_countdown.config(text=str(i)); time.sleep(1)
        self.lbl_countdown.config(text="¡YA!"); self.status_label.config(text="¡Grabando! Realiza la seña ahora."); time.sleep(1); self.lbl_countdown.place_forget()
        self.is_recording = True; self.recorded_frames = []; time.sleep(self.duration.get()); self.is_recording = False
        self.status_label.config(text="Procesando tu seña..."); self.lbl_result.config(text="Procesando...", background=self.PALETTE["warning"], foreground="white"); self.predict_and_show_result()
        self.btn_practice.config(state="normal"); self.combo_word.config(state="readonly")

    def _update_canvas(self, canvas, frame, width, height):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img = Image.fromarray(img).resize((width, height), Image.LANCZOS); imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk; canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

if __name__ == '__main__':
    print("ERROR: Este script es un módulo y no debe ser ejecutado directamente.")
    print("Por favor, ejecute 'test_word_interface.py' y use el botón 'Modo Entrenamiento'.")