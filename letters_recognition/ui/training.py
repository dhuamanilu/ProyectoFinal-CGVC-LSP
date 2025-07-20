# lsp_app/ui/training.py
import os
import random
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from letters_recognition.common import PALETTE
from letters_recognition.common.assets import DATASET_DIR
import letters_recognition.common.vision as cv
from .base import BaseCameraWindow


class TrainingModeWindow(BaseCameraWindow):
    """Modo Entrenamiento - LSP"""
    def __init__(self, parent):
        super().__init__(parent, "Modo Entrenamiento – LSP")

        self.target_letter = None
        self.correct_counter = 0
        self.total_counter = 0
        self.examples = []
        self.example_idx = 0

        self._init_styles()
        self._build_ui()

    # ---------- estilos ----------
    def _init_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("TLabel", background=PALETTE["bg"])
        style.configure("Green.Horizontal.TProgressbar",
                        background=PALETTE["secondary"])
        style.configure("Yellow.Horizontal.TProgressbar",
                        background=PALETTE["warning"])
        style.configure("Red.Horizontal.TProgressbar",
                        background=PALETTE["danger"])

    # ---------- UI ----------
    def _build_ui(self):
        # header
        header = tk.Frame(self, bg=PALETTE["primary"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="🎯 Modo Entrenamiento – LSP",
                 bg=PALETTE["primary"], fg="white",
                 font=("Arial", 18, "bold")).pack(pady=15)

        # controles superiores
        top = tk.Frame(self, bg=PALETTE["bg"])
        top.pack(fill="x", pady=(5, 0))
        self.btn_practice = tk.Button(
            top, text="Iniciar Práctica", command=self.toggle_practice,
            bg=PALETTE["secondary"], fg="white", font=("Arial", 11, "bold"))
        self.btn_practice.pack(side="left", padx=10)
        tk.Button(
            top, text="Volver", command=self.close,
            bg=PALETTE["danger"], fg="white", font=("Arial", 11, "bold")
        ).pack(side="left")

        # content
        content = tk.Frame(self, bg=PALETTE["bg"])
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # ------- cámara (izq) -------
        left = tk.Frame(content, bg=PALETTE["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cam_wrap = tk.Frame(left, width=self.CAM_W, height=self.CAM_H,
                            bg=PALETTE["border"], bd=2, relief="solid")
        cam_wrap.pack(pady=5)
        cam_wrap.pack_propagate(False)

        self.canvas = tk.Canvas(cam_wrap, width=self.CAM_W, height=self.CAM_H,
                                bg=PALETTE["light_bg"])
        self.canvas.pack()
        self._show_loading_camera()

        # ------- panel derecho -------
        right = tk.Frame(content, width=350, bg=PALETTE["bg"])
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # selector de letra
        sel = tk.LabelFrame(right, text="Seleccionar Letra",
                            bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        sel.pack(fill="x", pady=5)

        letters = [l.upper() for l in sorted(os.listdir(DATASET_DIR))
                   if len(l) == 1 and l.isalpha()]
        self.combo = ttk.Combobox(sel, state="readonly",
                                  values=letters, width=8,
                                  font=("Arial", 14), justify="center")
        self.combo.pack(pady=10)
        self.combo.bind("<<ComboboxSelected>>", self._on_letter_select)

        # letra objetivo
        target = tk.LabelFrame(right, text="Letra Objetivo",
                               bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        target.pack(fill="x", pady=5)
        self.lbl_obj = tk.Label(target, text="-", width=3, height=1,
                                font=("Arial", 48, "bold"),
                                bg=PALETTE["light_bg"],
                                fg=PALETTE["primary"], relief="solid", bd=1)
        self.lbl_obj.pack(pady=10)

        # progreso
        prog = tk.LabelFrame(right, text="Precisión",
                             bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        prog.pack(fill="x", pady=5)
        self.prec_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            prog, variable=self.prec_var, maximum=100, length=280,
            style="Red.Horizontal.TProgressbar")
        self.progress.pack(padx=10, pady=5)
        self.lbl_prec = tk.Label(prog, text="0% de precisión",
                                 bg=PALETTE["bg"], font=("Arial", 11))
        self.lbl_prec.pack(pady=5)

        # ejemplos
        ex = tk.LabelFrame(right, text="Ejemplos",
                           bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        ex.pack(fill="x", pady=5)
        self.example_pic = tk.Label(ex, bg=PALETTE["light_bg"],
                                    relief="solid", bd=1)
        self.example_pic.pack(padx=10, pady=5)
        nav = tk.Frame(ex, bg=PALETTE["bg"])
        nav.pack(pady=5)
        tk.Button(nav, text="◀", width=4, command=self.prev_example,
                  bg=PALETTE["gray"], fg="white").pack(side="left", padx=2)
        tk.Button(nav, text="▶", width=4, command=self.next_example,
                  bg=PALETTE["gray"], fg="white").pack(side="left", padx=2)

        # status bar
        self.status = tk.Label(self, text="Selecciona una letra para comenzar",
                               anchor="w", bg=PALETTE["light_bg"],
                               relief="sunken")
        self.status.pack(fill="x")

    # ---------- helpers ----------
    def _show_loading_camera(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.CAM_W, self.CAM_H,
                                     fill=PALETTE["light_bg"], outline="")
        self.canvas.create_text(self.CAM_W // 2, self.CAM_H // 2,
                                text="📷 Cámara lista para entrenamiento",
                                font=("Arial", 12), fill=PALETTE["gray"])

    def _on_letter_select(self, *_):
        letra = self.combo.get().lower()
        if letra:
            self.set_target_letter(letra)

    def set_target_letter(self, letter: str):
        self.target_letter = letter
        self.lbl_obj.config(text=letter.upper())
        self.correct_counter = self.total_counter = 0
        self.prec_var.set(0); self.lbl_prec.config(text="0% de precisión")
        self._update_progress_style(0)
        self.status.config(text=f"Letra '{letter.upper()}' seleccionada")
        self.load_examples()

    def load_examples(self):
        self.examples.clear()
        ruta = os.path.join(DATASET_DIR, self.target_letter)
        if os.path.exists(ruta):
            self.examples = [os.path.join(ruta, f)
                             for f in os.listdir(ruta)
                             if f.lower().endswith((".jpg", ".png"))]
            random.shuffle(self.examples)
            self.example_idx = 0
            self._show_example()

    def _show_example(self):
        if not self.examples:
            self.example_pic.config(image="", text="Sin ejemplos")
            return
        img = Image.open(self.examples[self.example_idx]).resize((150, 150))
        self.example_imgtk = ImageTk.PhotoImage(img)
        self.example_pic.config(image=self.example_imgtk, text="")

    def next_example(self):
        if self.examples:
            self.example_idx = (self.example_idx + 1) % len(self.examples)
            self._show_example()

    def prev_example(self):
        if self.examples:
            self.example_idx = (self.example_idx - 1) % len(self.examples)
            self._show_example()

    # ---------- práctica ----------
    def toggle_practice(self):
        if not self.target_letter:
            messagebox.showinfo("Selecciona una letra",
                                "Debes escoger una letra antes de practicar.")
            return
        if not self.running:
            self.btn_practice.config(text="Detener Práctica")
            self.status.config(text="Práctica en curso…")
            self._start_camera()
        else:
            self.btn_practice.config(text="Iniciar Práctica")
            self.status.config(text="Práctica detenida")
            self._stop_camera(); self._show_loading_camera()

    def _update_progress_style(self, val):
        if val <= 35:
            self.progress.config(style="Red.Horizontal.TProgressbar")
        elif val <= 70:
            self.progress.config(style="Yellow.Horizontal.TProgressbar")
        else:
            self.progress.config(style="Green.Horizontal.TProgressbar")

    # ---------- override cámara ----------
    def _update_frame(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = cv.hands.process(rgb)
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    cv.mp_drawing.draw_landmarks(frame, lm,
                                                cv.mp_hands.HAND_CONNECTIONS)

            if res.multi_hand_landmarks:
                coords = [coord
                        for lm in res.multi_hand_landmarks[0].landmark
                        for coord in (lm.x, lm.y)]
                if len(coords) == 42:
                    pred = cv.model.predict(np.array(coords).reshape(1, -1))[0]
                    letter = cv.label_encoder.inverse_transform([pred])[0]
                    self.total_counter += 1
                    if letter.lower() == self.target_letter:
                        self.correct_counter += 1
                    precision = (self.correct_counter /
                                self.total_counter) * 100
                    self.prec_var.set(precision)
                    self.lbl_prec.config(text=f"{precision:.0f}% de precisión")
                    self._update_progress_style(precision)

            imgtk = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((self.CAM_W, self.CAM_H)))

            if self.canvas.winfo_exists():
                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
                self.canvas.imgtk = imgtk
            else:
                self.running = False
                break

        self.after(0, self._safe_cleanup)

    def _safe_cleanup(self):
        self._stop_camera()
        if self.canvas.winfo_exists():
            self._show_loading_camera()
