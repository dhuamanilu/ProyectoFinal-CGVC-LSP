import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import numpy as np
import joblib
import os
import random

from letters_recognition.config import MODEL_PATH, ENCODER_PATH, DATASET_DIR

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

class TrainingModeWindow:
    CAM_WIDTH = 420
    CAM_HEIGHT = 320
    PALETTE = {"primary": "#4285f4", "danger": "#ea4335", "success": "#28a745", "warning": "#ff9f0a", "gray": "#6c757d", "bg": "#ffffff", "header": "#ff6b6b"}
    def __init__(self, parent):
        self.parent = parent
        self.target_letter = None
        self.running = False
        self.correct_counter = 0
        self.total_counter = 0
        self.examples = []
        self.example_idx = 0
        self.top = tk.Toplevel(parent.root)
        self.top.title("Modo Entrenamiento – LSP")
        self.top.geometry("880x640")
        self.top.resizable(False, False)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._init_styles()
        self._build_ui()
        self.cap = None
        self.frame_actual = None
    def _init_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.PALETTE["bg"])
        style.configure("TLabel", background=self.PALETTE["bg"])
        style.configure("TButton", font=("Arial", 10, "bold"))
        style.configure("Pic.TLabel", relief="solid", borderwidth=1, anchor="center")
        style.configure("Red.Horizontal.TProgressbar", troughcolor="#dee2e6", bordercolor="#dee2e6", background=self.PALETTE["danger"], thickness=16)
        style.configure("Yellow.Horizontal.TProgressbar", troughcolor="#dee2e6", bordercolor="#dee2e6", background=self.PALETTE["warning"], thickness=16)
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#dee2e6", bordercolor="#dee2e6", background=self.PALETTE["success"], thickness=16)
    def _build_ui(self):
        header = ttk.Frame(self.top, height=50)
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Label(header, text="Modo Entrenamiento – Lengua de Señas Peruana", background=self.PALETTE["header"], foreground="white", font=("Arial", 16, "bold"), anchor="center").pack(fill="both", expand=True)
        cam_frame = ttk.Frame(self.top, width=self.CAM_WIDTH, height=self.CAM_HEIGHT)
        cam_frame.grid(row=1, column=0, padx=10, pady=10)
        cam_frame.grid_propagate(False)
        self.canvas = tk.Canvas(cam_frame, width=self.CAM_WIDTH, height=self.CAM_HEIGHT, bg="#e9ecef", bd=0, highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(self.CAM_WIDTH // 2, self.CAM_HEIGHT // 2, text="Tu señal", font=("Arial", 14, "italic"), fill=self.PALETTE["gray"])
        feedback = ttk.Frame(self.top, width=260, height=self.CAM_HEIGHT)
        feedback.grid(row=1, column=1, sticky="n", pady=10)
        feedback.grid_propagate(False)
        ttk.Label(feedback, text="Realiza esta letra", font=("Arial", 12, "bold")).pack(pady=(8, 4))
        self.lbl_objetivo = ttk.Label(feedback, text="-", width=4, font=("Arial", 60), style="Pic.TLabel")
        self.lbl_objetivo.pack(pady=(0, 18))
        self.precision_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(feedback, variable=self.precision_var, style="Red.Horizontal.TProgressbar", maximum=100, length=220, mode="determinate")
        self.progress.pack(padx=10)
        self.lbl_precision = ttk.Label(feedback, text="0 % de precisión")
        self.lbl_precision.pack(pady=(4, 12))
        gallery = ttk.LabelFrame(feedback, text="Ejemplos")
        gallery.pack(fill="x", padx=8)
        self.example_pic = ttk.Label(gallery, image=None, style="Pic.TLabel")
        self.example_pic.pack(padx=4, pady=4)
        nav = ttk.Frame(gallery)
        nav.pack(pady=3)
        ttk.Button(nav, text="◀", width=4, command=self.prev_example).pack(side="left", padx=2)
        ttk.Button(nav, text="▶", width=4, command=self.next_example).pack(side="left", padx=2)
        selector_frame = ttk.Frame(self.top)
        selector_frame.grid(row=1, column=2, sticky="nsw", padx=5, pady=10)
        ttk.Label(selector_frame, text="Letras disponibles", font=("Arial", 11, "bold")).pack(anchor="w")
        letras = [l.upper() for l in sorted(os.listdir(DATASET_DIR)) if len(l) == 1 and l.isalpha()]
        self.combo = ttk.Combobox(selector_frame, state="readonly", values=letras, font=("Arial", 11), width=5)
        self.combo.pack()
        self.combo.bind("<<ComboboxSelected>>", self._on_letter_select)
        controls = ttk.Frame(self.top)
        controls.grid(row=2, column=0, columnspan=3, pady=10)
        self.btn_practice = ttk.Button(controls, text="Iniciar práctica", width=16, command=self.toggle_practice)
        self.btn_practice.grid(row=0, column=0, padx=6)
        ttk.Button(controls, text="Volver", width=16, command=self.close).grid(row=0, column=1, padx=6)
        self.status = ttk.Label(self.top, text="Estado: selecciona una letra", anchor="w", relief="sunken")
        self.status.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.top.columnconfigure(0, weight=1)
    def _on_letter_select(self, *_):
        letra = self.combo.get().lower()
        if letra:
            self.set_target_letter(letra)
    def set_target_letter(self, letter):
        self.target_letter = letter
        self.lbl_objetivo.config(text=letter.upper())
        self.precision_var.set(0)
        self.lbl_precision.config(text="0 % de precisión")
        self.correct_counter = 0
        self.total_counter = 0
        self.examples = []
        self.example_pic.config(image="")
        self.status.config(text=f"Letra objetivo «{letter.upper()}» seleccionada")
        self.load_examples()
    def toggle_practice(self):
        if not self.target_letter:
            messagebox.showinfo("Selecciona una letra", "Debes escoger una letra antes de practicar.")
            return
        if not self.running:
            self.running = True
            self.btn_practice.config(text="Detener práctica")
            self.cap = cv2.VideoCapture(0)
            self.status.config(text="Práctica en curso…")
            threading.Thread(target=self._update_frame, daemon=True).start()
        else:
            self.running = False
            self.btn_practice.config(text="Iniciar práctica")
            self._stop_camera()
    def update_progressbar_color(self, precision):
        if precision <= 35:
            self.progress.config(style="Red.Horizontal.TProgressbar")
        elif precision <= 70:
            self.progress.config(style="Yellow.Horizontal.TProgressbar")
        else:
            self.progress.config(style="Green.Horizontal.TProgressbar")
    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            predicted_letter = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                    if len(landmarks) == 42:
                        X = np.array(landmarks).reshape(1, -1)
                        pred = model.predict(X)[0]
                        predicted_letter = label_encoder.inverse_transform([pred])[0]
            if predicted_letter:
                self.total_counter += 1
                if predicted_letter.lower() == self.target_letter:
                    self.correct_counter += 1
                precision = (self.correct_counter / self.total_counter) * 100
                self.precision_var.set(precision)
                self.lbl_precision.config(text=f"{precision:.0f} % de precisión")
                self.update_progressbar_color(precision)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((self.CAM_WIDTH, self.CAM_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk
        self._stop_camera()
    def _stop_camera(self):
        if self.cap:
            self.cap.release()
        self.canvas.delete("all")
        self.canvas.create_text(self.CAM_WIDTH // 2, self.CAM_HEIGHT // 2, text="Tu señal", font=("Arial", 14, "italic"), fill=self.PALETTE["gray"])
        self.status.config(text="Práctica detenida")
    def load_examples(self):
        if not self.target_letter:
            return
        ruta = os.path.join(DATASET_DIR, self.target_letter)
        self.examples = [os.path.join(ruta, f) for f in os.listdir(ruta) if f.lower().endswith((".jpg", ".png"))]
        if not self.examples:
            return
        random.shuffle(self.examples)
        self.example_idx = 0
        self._show_example()
    def _show_example(self):
        if not self.examples:
            return
        path = self.examples[self.example_idx]
        img = Image.open(path).resize((150, 150))
        self.example_imgtk = ImageTk.PhotoImage(img)
        self.example_pic.config(image=self.example_imgtk)
    def next_example(self):
        if self.examples:
            self.example_idx = (self.example_idx + 1) % len(self.examples)
            self._show_example()
    def prev_example(self):
        if self.examples:
            self.example_idx = (self.example_idx - 1) % len(self.examples)
            self._show_example()
    def close(self):
        self.running = False
        self._stop_camera()
        self.top.destroy()
        self.parent.root.deiconify()

class LSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aprende LSP - Lengua de Señas Peruana")
        self.root.geometry("940x640")
        self.root.resizable(False, False)
        self.letra_actual = "-"
        self.palabra_formada = ""
        self.captura = None
        self.running = False
        self.frame_actual = None
        self._build_main_ui()
    def _build_main_ui(self):
        header = tk.Frame(self.root, bg="#4285f4", height=50)
        header.pack(fill=tk.X)
        tk.Label(header, text="Aprende LSP - Lengua de Señas Peruana", bg="#4285f4", fg="white", font=("Arial", 16, "bold")).pack(pady=8)
        content = tk.Frame(self.root, bg="white")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        cam_frame = tk.Frame(content, width=520, height=380, bg="#e9ecef")
        cam_frame.pack(side=tk.LEFT, padx=10)
        cam_frame.pack_propagate(False)
        self.canvas = tk.Canvas(cam_frame, width=520, height=380, bg="#e9ecef", bd=0, highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(260, 190, text="Cámara", font=("Arial", 14, "italic"), fill="#6c757d")
        right = tk.Frame(content, width=320, height=380, bg="white")
        right.pack(side=tk.LEFT, padx=10)
        right.pack_propagate(False)
        tk.Label(right, text="Letra Reconocida", font=("Arial", 13, "bold"), bg="white").pack(pady=(10, 4))
        self.lbl_letra = tk.Label(right, text=self.letra_actual, font=("Arial", 60), bg="#f8f9fa", width=3)
        self.lbl_letra.pack(pady=(0, 20))
        self.lbl_palabra = tk.Label(right, text=self.palabra_formada, font=("Arial", 24), bg="#f1f3f5", width=14)
        self.lbl_palabra.pack(pady=(0, 4))
        tk.Label(right, text="Palabra formada", font=("Arial", 10), bg="white").pack()
        controls = tk.Frame(self.root, bg="white")
        controls.pack(pady=12)
        tk.Button(controls, text="Iniciar", bg="#28a745", fg="white", font=("Arial", 10, "bold"), width=14, command=self.toggle_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Predecir Letra", bg="#ff9f0a", fg="white", font=("Arial", 10, "bold"), width=14, command=self.realizar_prediccion).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Borrar Letra", bg="#4285f4", fg="white", font=("Arial", 10, "bold"), width=14, command=self.borrar_letra).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Borrar Todo", bg="#4285f4", fg="white", font=("Arial", 10, "bold"), width=14, command=self.borrar_todo).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Entrenamiento", bg="#ff6b6b", fg="white", font=("Arial", 10, "bold"), width=14, command=self.abrir_modo_entrenamiento).pack(side=tk.LEFT, padx=5)
        self.estado = tk.Label(self.root, text="Estado: Listo", anchor=tk.W, bg="#f1f3f5")
        self.estado.pack(fill=tk.X)
    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.captura = cv2.VideoCapture(0)
            self.estado.config(text="Estado: Reconocimiento activo")
            threading.Thread(target=self._actualizar_frame, daemon=True).start()
        else:
            self.running = False
            if self.captura:
                self.captura.release()
            self.canvas.delete("all")
            self.estado.config(text="Estado: Listo")
    def _actualizar_frame(self):
        while self.running:
            ret, frame = self.captura.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((520, 380))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
            self.frame_actual = frame.copy()
    def realizar_prediccion(self):
        if self.frame_actual is None:
            return
        frame = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                if len(landmarks) == 42:
                    X = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(X)[0]
                    letter = label_encoder.inverse_transform([pred])[0]
                    self.letra_actual = letter.upper()
                    self.lbl_letra.config(text=self.letra_actual)
                    self.palabra_formada += self.letra_actual
                    self.lbl_palabra.config(text=self.palabra_formada)
    def borrar_letra(self):
        if self.palabra_formada:
            self.palabra_formada = self.palabra_formada[:-1]
            self.lbl_palabra.config(text=self.palabra_formada)
    def borrar_todo(self):
        self.palabra_formada = ""
        self.lbl_palabra.config(text="")
        self.lbl_letra.config(text="-")
    def abrir_modo_entrenamiento(self):
        self.root.withdraw()
        TrainingModeWindow(self)

if __name__ == "__main__":
    root = tk.Tk()
    app = LSPApp(root)
    root.mainloop()