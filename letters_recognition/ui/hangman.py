# lsp_app/ui/hangman.py
import random, numpy as np, cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from letters_recognition.common import PALETTE
import letters_recognition.common.vision as cv
from .base import BaseCameraWindow


class HangmanModeWindow(BaseCameraWindow):
    WORD_LIST = ["PERRO", "GATO", "CASA", "AMIGO",
                 "FAMILIA", "PYTHON", "AHORCADO"]
    MAX_ERRORS = 10

    def __init__(self, parent):
        super().__init__(parent, "Modo Ahorcado – LSP")
        self.word = ""; self.guessed = set(); self.errors = 0
        self._build_ui(); self.reset_game()

    # ---------- UI ----------
    def _build_ui(self):
        hdr = tk.Frame(self, bg=PALETTE["primary"], height=60)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🔤 Modo Ahorcado",
                 bg=PALETTE["primary"], fg="white",
                 font=("Arial", 18, "bold")).pack(pady=15)

        content = tk.Frame(self, bg=PALETTE["bg"])
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # cámara
        left = tk.Frame(content, bg=PALETTE["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        cam_wrap = tk.Frame(left, width=self.CAM_W, height=self.CAM_H,
                            bg=PALETTE["border"], bd=2, relief="solid")
        cam_wrap.pack(pady=5); cam_wrap.pack_propagate(False)
        self.canvas = tk.Canvas(cam_wrap, width=self.CAM_W, height=self.CAM_H,
                                bg=PALETTE["light_bg"])
        self.canvas.pack()
        self._show_placeholder()

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
        right = tk.Frame(content, width=300, bg=PALETTE["bg"])
        right.pack(side="right", fill="y"); right.pack_propagate(False)

        pf = tk.LabelFrame(right, text="Palabra",
                           bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        pf.pack(fill="x", pady=5)
        self.lbl_word = tk.Label(pf, text="", height=2, width=15,
                                 font=("Arial", 24, "bold"),
                                 bg=PALETTE["light_bg"], relief="solid", bd=1)
        self.lbl_word.pack(padx=10, pady=10)

        lf = tk.LabelFrame(right, text="Letras Intentadas",
                           bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        lf.pack(fill="x", pady=5)
        self.lbl_tried = tk.Label(lf, text="", height=2, wraplength=280,
                                  font=("Arial", 14),
                                  bg=PALETTE["light_bg"])
        self.lbl_tried.pack(padx=10, pady=10)

        hf = tk.LabelFrame(right, text="Ahorcado",
                           bg=PALETTE["bg"], font=("Arial", 12, "bold"))
        hf.pack(fill="both", expand=True, pady=5)
        self.hang_canvas = tk.Canvas(hf, width=200, height=240,
                                     bg=PALETTE["light_bg"])
        self.hang_canvas.pack(padx=10, pady=10)

        ctl = tk.Frame(self, bg=PALETTE["bg"])
        ctl.pack(fill="x", pady=(0, 10))
        tk.Button(ctl, text="Reiniciar", command=self.reset_game,
                  bg=PALETTE["secondary"], fg="white",
                  font=("Arial", 11, "bold")).pack(side="left",
                                                   expand=True, fill="x", padx=5)
        tk.Button(ctl, text="Volver", command=self.close,
                  bg=PALETTE["danger"], fg="white",
                  font=("Arial", 11, "bold")).pack(side="left",
                                                   expand=True, fill="x", padx=5)

    # ---------- helpers ----------
    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.CAM_W, self.CAM_H,
                                     fill=PALETTE["light_bg"], outline="")
        self.canvas.create_text(self.CAM_W // 2, self.CAM_H // 2,
                                text="📷 Cámara lista\nHaz clic en 'Iniciar Cámara'",
                                font=("Arial", 12), fill=PALETTE["gray"])

    # ---------- lógica ----------
    def reset_game(self):
        self.word = random.choice(self.WORD_LIST)
        self.guessed.clear(); self.errors = 0
        self._update_word_display(); self._update_tried(); self._draw_hangman()

    def _update_word_display(self):
        self.lbl_word.config(text=" ".join(
            c if c in self.guessed else "_" for c in self.word))

    def _update_tried(self):
        self.lbl_tried.config(text=" ".join(sorted(self.guessed)))

    def _draw_hangman(self):
        self.hang_canvas.delete("all")
        parts = [
            lambda c: c.create_line(20, 220, 180, 220),  # suelo
            lambda c: c.create_line(50, 220, 50, 20),    # poste
            lambda c: c.create_line(50, 20, 140, 20),    # viga
            lambda c: c.create_line(140, 20, 140, 50),   # cuerda
            lambda c: c.create_oval(120, 50, 160, 90),   # cabeza
            lambda c: c.create_line(140, 90, 140, 150),  # cuerpo
            lambda c: c.create_line(140, 110, 120, 130), # brazo izq
            lambda c: c.create_line(140, 110, 160, 130), # brazo der
            lambda c: c.create_line(140, 150, 120, 180), # pierna izq
            lambda c: c.create_line(140, 150, 160, 180), # pierna der
        ]
        for i in range(min(self.errors, self.MAX_ERRORS)):
            parts[i](self.hang_canvas)

    # ---------- cámara ----------
    def toggle_camera(self):
        if not self.running:
            self.btn_cam.config(text="Detener Cámara")
            self._start_camera()
        else:
            self.btn_cam.config(text="Iniciar Cámara")
            self._stop_camera(); self._show_placeholder()

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
                        cv.mp_drawing.draw_landmarks(frame, lm,
                                                    cv.mp_hands.HAND_CONNECTIONS)

                if not self.running or not self.canvas.winfo_exists():
                    break

                imgtk = ImageTk.PhotoImage(Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((self.CAM_W, self.CAM_H)))

                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
                self.canvas.imgtk = imgtk
                self.frame_actual = frame.copy()

        except tk.TclError:
            pass  # puede lanzarse si la ventana fue cerrada mientras se actualizaba

        # Limpiar cámara y mostrar placeholder desde el hilo principal
        self.after(0, self._safe_cleanup)

    def _safe_cleanup(self):
        self._stop_camera()
        if self.canvas.winfo_exists():
            self._show_placeholder()


    # ---------- reconocimiento ----------
    def recognize_letter(self):
        if self.frame_actual is None:
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
        letter = cv.label_encoder.inverse_transform(
            [cv.model.predict(np.array(coords).reshape(1, -1))[0]])[0].upper()
        if letter in self.guessed:
            messagebox.showinfo("Letra repetida", f"Ya probaste '{letter}'.")
            return
        self.guessed.add(letter)
        if letter in self.word:
            self._update_word_display()
            if all(c in self.guessed for c in self.word):
                messagebox.showinfo("¡Ganaste!", f"La palabra era '{self.word}'.")
        else:
            self.errors += 1; self._draw_hangman()
            if self.errors >= self.MAX_ERRORS:
                messagebox.showinfo("Perdiste", f"La palabra era '{self.word}'.")
        self._update_tried()
