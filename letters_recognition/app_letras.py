# interfaz_tkinter_lsp.py – GUI con predicción por botón, sin parpadeo y landmarks activados

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import numpy as np
import joblib
import time
import os
import random
from letters_recognition.config import  MODEL_PATH, ENCODER_PATH , DATASET_DIR
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5)

class LSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aprende LSP - Lengua de Señas Peruana")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        self.letra_actual = "-"
        self.palabra_formada = ""
        self.captura = None
        self.running = False
        self.modo_entrenamiento = False
        self.dataset_path = DATASET_DIR
        self.frame_actual = None

        self._build_main_ui()

    def _build_main_ui(self):
        self.header = tk.Frame(self.root, bg="#4285f4", height=40)
        self.header.pack(fill=tk.X)
        tk.Label(self.header, text="Aprende LSP - Lengua de Señas Peruana", bg="#4285f4", fg="white",
                 font=("Arial", 14, "bold")).pack(pady=5)

        self.content = tk.Frame(self.root, bg="white")
        self.content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_panel = tk.Frame(self.content, width=500, height=400, bg="#f0f2f5")
        self.left_panel.pack(side=tk.LEFT, padx=10)
        self.left_panel.pack_propagate(False)
        self.canvas = tk.Canvas(self.left_panel, width=500, height=400, bg="#f0f2f5", bd=0, highlightthickness=0)
        self.canvas.pack()

        self.right_panel = tk.Frame(self.content, width=300, height=400, bg="white")
        self.right_panel.pack(side=tk.LEFT, padx=10)

        tk.Label(self.right_panel, text="Letra Reconocida", font=("Arial", 12, "bold"), bg="white").pack(pady=10)
        self.lbl_letra = tk.Label(self.right_panel, text=self.letra_actual, font=("Arial", 40), width=4, bg="#f5f5f5")
        self.lbl_letra.pack(pady=10)

        self.lbl_palabra = tk.Label(self.right_panel, text=self.palabra_formada, font=("Arial", 20), bg="#e6e6e6", width=15)
        self.lbl_palabra.pack(pady=20)
        tk.Label(self.right_panel, text="Palabra formada", font=("Arial", 10), bg="white").pack()

        self.controls = tk.Frame(self.root, bg="white")
        self.controls.pack(pady=10)
        tk.Button(self.controls, text="Iniciar", bg="green", fg="white", font=("Arial", 10), width=12,
                  command=self.toggle_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls, text="Predecir Letra", bg="orange", fg="white", font=("Arial", 10), width=12,
                  command=self.realizar_prediccion).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls, text="Borrar Letra", bg="#4285f4", fg="white", font=("Arial", 10), width=12,
                  command=self.borrar_letra).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls, text="Borrar Todo", bg="#4285f4", fg="white", font=("Arial", 10), width=12,
                  command=self.borrar_todo).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls, text="Entrenamiento", bg="tomato", fg="white", font=("Arial", 10), width=12,
                  command=self.abrir_modo_entrenamiento).pack(side=tk.LEFT, padx=5)

        self.estado = tk.Label(self.root, text="Estado: Listo", anchor=tk.W)
        self.estado.pack(fill=tk.X, side=tk.LEFT)

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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
            self.frame_actual = frame.copy()

    def realizar_prediccion(self):
        if not hasattr(self, 'frame_actual'):
            return

        frame = cv2.cvtColor(self.frame_actual, cv2.COLOR_RGB2BGR)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                if len(landmarks) == 42:
                    X = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(X)[0]
                    letter = label_encoder.inverse_transform([prediction])[0]
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
        top = tk.Toplevel(self.root)
        top.title("Modo Entrenamiento")
        top.geometry("300x380")

        letras = sorted(os.listdir(self.dataset_path))
        self.img_label = tk.Label(top)
        self.img_label.pack(pady=10)

        def mostrar_ejemplo(letra):
            ruta = os.path.join(self.dataset_path, letra)
            ejemplos = [f for f in os.listdir(ruta) if f.endswith('.jpg') or f.endswith('.png')]
            if ejemplos:
                archivo = random.choice(ejemplos)
                img = Image.open(os.path.join(ruta, archivo))
                img = img.resize((200, 200))
                img_tk = ImageTk.PhotoImage(img)
                self.img_label.config(image=img_tk)
                self.img_label.image = img_tk
                tk.Label(top, text=f"Letra: {letra.upper()}", font=("Arial", 14)).pack()

        combo = ttk.Combobox(top, values=letras, state="readonly")
        combo.pack(pady=10)
        combo.bind("<<ComboboxSelected>>", lambda e: mostrar_ejemplo(combo.get()))

        tk.Button(top, text="Cerrar", command=top.destroy).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = LSPApp(root)
    root.mainloop()
