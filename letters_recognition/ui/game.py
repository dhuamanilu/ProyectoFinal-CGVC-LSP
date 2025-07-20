# lsp_app/ui/game.py
import os, random, numpy as np, cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from letters_recognition.common import PALETTE
from letters_recognition.common.assets import DATASET_DIR
import letters_recognition.common.vision as cv
from .base import BaseCameraWindow


class GameModeWindow(BaseCameraWindow):
    """Modo Juego – Deletrea tu Nombre"""
    def __init__(self, parent):
        super().__init__(parent, "Modo Juego – Deletrea tu Nombre")

        self.target_name = ""
        self.recognized_letters = []
        self.current_letter_idx = 0
        self.examples = []; self.example_idx = 0

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        header = tk.Frame(self, bg=PALETTE["primary"], height=60)
        header.pack(fill="x"); header.pack_propagate(False)
        tk.Label(header, text="🎮 Modo Juego – Deletrea tu Nombre",
                 bg=PALETTE["primary"], fg="white",
                 font=("Arial", 18, "bold")).pack(pady=15)

        content = tk.Frame(self, bg=PALETTE["bg"])
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # entrada de nombre
        name_fr = tk.Frame(content, bg=PALETTE["light_bg"], relief="solid", bd=1)
        name_fr.pack(fill="x", pady=(0, 15))
        tk.Label(name_fr, text="Ingresa tu nombre:",
                 bg=PALETTE["light_bg"], font=("Arial", 12, "bold")).pack(pady=5)
        ef = tk.Frame(name_fr, bg=PALETTE["light_bg"]); ef.pack(pady=5)
        self.entry_name = tk.Entry(ef, font=("Arial", 14), width=20, justify="center")
        self.entry_name.pack(side="left", padx=5)
        tk.Button(ef, text="Empezar Juego",
                  command=self.start_game,
                  bg=PALETTE["secondary"], fg="white",
                  font=("Arial", 11, "bold")).pack(side="left", padx=5)

        # layout principal
        main = tk.Frame(content, bg=PALETTE["bg"])
        main.pack(fill="both", expand=True)

        # cámara
        left = tk.Frame(main, bg=PALETTE["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        cam_wrap = tk.Frame(left, width=self.CAM_W, height=self.CAM_H,
                            bg=PALETTE["border"], bd=2, relief="solid")
        cam_wrap.pack(pady=5); cam_wrap.pack_propagate(False)
        self.canvas = tk.Canvas(cam_wrap, width=self.CAM_W, height=self.CAM_H,
                                bg=PALETTE["light_bg"])
        self.canvas.pack()
        self._show_cam_placeholder()
        cam_ctrl = tk.Frame(left, bg=PALETTE["bg"]); cam_ctrl.pack(pady=10)
        self.btn_cam = tk.Button(cam_ctrl, text="Iniciar Cámara",
                                 command=self.toggle_camera,
                                 bg=PALETTE["primary"], fg="white",
                                 font=("Arial", 11, "bold"))
        self.btn_cam.pack(side="left", padx=5)
        tk.Button(cam_ctrl, text="Reconocer Letra",
                  command=self.recognize_letter,
                  bg=PALETTE["warning"], fg="white",
                  font=("Arial", 11, "bold")).pack(side="left", padx=5)

        # panel derecho
        right = tk.Frame(main, width=350, bg=PALETTE["bg"])
        right.pack(side="right", fill="y"); right.pack_propagate(False)

        prog = tk.LabelFrame(right, text="Progreso del Juego",
                             bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        prog.pack(fill="x", pady=5)
        self.lbl_display = tk.Label(
            prog, text="", height=2,
            font=("Arial", 16, "bold"),
            bg=PALETTE["light_bg"], relief="solid", bd=1)
        self.lbl_display.pack(fill="x", padx=10, pady=5)

        cur = tk.LabelFrame(right, text="Letra Actual",
                            bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        cur.pack(fill="x", pady=5)
        self.lbl_cur = tk.Label(cur, text="-", width=3, height=1,
                                font=("Arial", 48, "bold"),
                                bg=PALETTE["light_bg"],
                                fg=PALETTE["primary"], relief="solid", bd=1)
        self.lbl_cur.pack(pady=10)

        ex = tk.LabelFrame(right, text="Ejemplo de la Seña",
                           bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        ex.pack(fill="x", pady=5)
        self.example_pic = tk.Label(ex, bg=PALETTE["light_bg"],
                                    relief="solid", bd=1)
        self.example_pic.pack(padx=10, pady=5)

        # controles inferiores
        bottom = tk.Frame(content, bg=PALETTE["bg"])
        bottom.pack(fill="x", pady=(10, 0))
        for txt, cmd, color in (
            ("Volver", self.close, PALETTE["danger"]),
            ("Reiniciar Juego", self.restart_game, PALETTE["warning"]),
            ("Siguiente Letra", self.next_letter, PALETTE["secondary"]),
        ):
            tk.Button(bottom, text=txt, command=cmd,
                      bg=color, fg="white",
                      font=("Arial", 11, "bold")).pack(
                          side="left", expand=True, fill="x", padx=5)

        self.status = tk.Label(self, text="Ingresa tu nombre para comenzar",
                               anchor="w", bg=PALETTE["light_bg"],
                               relief="sunken")
        self.status.pack(fill="x")

    # ---------- helpers ----------
    def _show_cam_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.CAM_W, self.CAM_H,
                                     fill=PALETTE["light_bg"], outline="")
        self.canvas.create_text(self.CAM_W // 2, self.CAM_H // 2,
                                text="📷 Cámara lista\nHaz clic en 'Iniciar Cámara'",
                                font=("Arial", 12), fill=PALETTE["gray"])

    # ---------- juego ----------
    def start_game(self):
        name = self.entry_name.get().strip().upper()
        if not name or not name.isalpha():
            messagebox.showwarning("Nombre inválido",
                                   "Ingresa un nombre válido (solo letras).")
            return
        self.target_name = name
        self.recognized_letters = ["_"] * len(name)
        self.current_letter_idx = 0
        self._update_display()
        self._load_examples()
        self.status.config(text=f"¡Juego iniciado! Deletrea: {name}")

    def _update_display(self):
        if not self.target_name:
            return
        txt = " ".join(
            f"[{self.recognized_letters[i]}]" if i == self.current_letter_idx
            else f"{self.recognized_letters[i]}"
            for i in range(len(self.target_name)))
        self.lbl_display.config(text=txt)
        if self.current_letter_idx < len(self.target_name):
            self.lbl_cur.config(text=self.target_name[self.current_letter_idx])
        else:
            self.lbl_cur.config(text="✓")

    def _load_examples(self):
        self.examples.clear()
        if self.current_letter_idx >= len(self.target_name):
            return
        letter = self.target_name[self.current_letter_idx].lower()
        ruta = os.path.join(DATASET_DIR, letter)
        if os.path.exists(ruta):
            self.examples = [os.path.join(ruta, f)
                             for f in os.listdir(ruta)
                             if f.lower().endswith((".jpg", ".png"))]
            random.shuffle(self.examples); self.example_idx = 0
            self._show_example()

    def _show_example(self):
        if not self.examples:
            self.example_pic.config(image="", text="Sin ejemplos")
            return
        img = Image.open(self.examples[self.example_idx]).resize((150, 150))
        self.example_imgtk = ImageTk.PhotoImage(img)
        self.example_pic.config(image=self.example_imgtk, text="")

    def next_letter(self):
        if self.target_name and self.current_letter_idx < len(self.target_name) - 1:
            self.current_letter_idx += 1
            self._load_examples(); self._update_display()

    def restart_game(self):
        if self.target_name:
            self.current_letter_idx = 0
            self.recognized_letters = ["_"] * len(self.target_name)
            self._load_examples(); self._update_display()
            self.status.config(text=f"Juego reiniciado – Deletrea: {self.target_name}")

    # ---------- cámara ----------
    def toggle_camera(self):
        if not self.running:
            self.btn_cam.config(text="Detener Cámara")
            self._start_camera()
        else:
            self.btn_cam.config(text="Iniciar Cámara")
            self._stop_camera(); self._show_cam_placeholder()

    def _update_frame(self):
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = cv.hands.process(rgb)
                if res.multi_hand_landmarks:
                    for lm in res.multi_hand_landmarks:
                        cv.mp_drawing.draw_landmarks(frame, lm, cv.mp_hands.HAND_CONNECTIONS)

                if not self.running or not self.canvas.winfo_exists():
                    break

                imgtk = ImageTk.PhotoImage(Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((self.CAM_W, self.CAM_H)))

                # actualizar la imagen en el canvas
                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
                self.canvas.imgtk = imgtk
                self.frame_actual = frame.copy()
        except tk.TclError:
            pass  # ocurre si la ventana se cerró mientras el hilo corría

        # detener la cámara desde hilo principal
        self.after(0, self._safe_cleanup)

    def _safe_cleanup(self):
        self._stop_camera()
        if self.canvas.winfo_exists():
            self._show_cam_placeholder()

    # ---------- reconocimiento ----------
    def recognize_letter(self):
        if not self.target_name or self.frame_actual is None:
            return
        rgb = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
        res = cv.hands.process(rgb)
        if not res.multi_hand_landmarks:
            messagebox.showinfo("Sin mano", "No se detectó mano.")
            return
        coords = [coord for lm in res.multi_hand_landmarks[0].landmark
                  for coord in (lm.x, lm.y)]
        if len(coords) != 42:
            return
        pred = cv.model.predict(np.array(coords).reshape(1, -1))[0]
        letter = cv.label_encoder.inverse_transform([pred])[0].upper()
        target_letter = self.target_name[self.current_letter_idx]
        if letter == target_letter:
            self.recognized_letters[self.current_letter_idx] = letter
            self._update_display()
            messagebox.showinfo("¡Correcto!", f"Reconociste '{letter}'")
            if self.current_letter_idx < len(self.target_name) - 1:
                self.current_letter_idx += 1
                self._load_examples(); self._update_display()
            else:
                messagebox.showinfo("¡Felicidades!",
                                    f"Completaste tu nombre: {self.target_name}")
                self.status.config(text="¡Juego completado!")
        else:
            messagebox.showinfo("Intenta de nuevo",
                                f"Reconocí '{letter}', necesitas '{target_letter}'")
