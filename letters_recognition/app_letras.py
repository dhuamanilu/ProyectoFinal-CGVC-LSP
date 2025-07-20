# lsp_app/main.py
import threading, cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

from letters_recognition.common import PALETTE, async_init_vision
import letters_recognition.common.vision as cv
from letters_recognition.ui import TrainingModeWindow, GameModeWindow, HangmanModeWindow


class LoadingWindow:
    def __init__(self, on_ready):
        self.root = tk.Tk()
        self.root.title("LSP – Cargando…")
        self.root.geometry("400x180")
        self.root.resizable(False, False)
        self.root.configure(bg=PALETTE["primary"])

        ttk.Style(self.root).theme_use("clam")
        tk.Label(self.root, text="LSP – Lengua de Señas Peruana",
                 bg=PALETTE["primary"], fg="white",
                 font=("Arial", 16, "bold")).pack(pady=20)

        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=300)
        self.progress.pack(pady=15); self.progress.start(8)
        self.status = tk.Label(self.root, text="Cargando modelo...",
                               bg=PALETTE["primary"], fg="white", font=("Arial", 10))
        self.status.pack()

        async_init_vision(self.finish); self.on_ready = on_ready

    def finish(self):
        self.root.after(200, self._done)

    def _done(self):
        self.progress.stop()
        self.root.destroy()
        self.on_ready()

    def show(self):
        self.root.mainloop()


class LSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LSP – Lengua de Señas Peruana")
        self.root.geometry("980x720"); self.root.resizable(False, False)
        self.root.configure(bg=PALETTE["bg"])

        self.running = False; self.captura = None
        self.frame_actual = None; self.letra_actual = "-"
        self.palabra_formada = ""

        self._build_ui()

    # ---------- UI principal ----------
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=PALETTE["primary"], height=70)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🤟 LSP – Lengua de Señas Peruana",
                 bg=PALETTE["primary"], fg="white",
                 font=("Arial", 20, "bold")).pack(pady=20)

        content = tk.Frame(self.root, bg=PALETTE["bg"])
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # cámara
        left = tk.Frame(content, bg=PALETTE["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))
        cam_wrap = tk.Frame(left, width=540, height=400,
                            bg=PALETTE["border"], bd=2, relief="solid")
        cam_wrap.pack(pady=10); cam_wrap.pack_propagate(False)
        self.canvas = tk.Canvas(cam_wrap, width=540, height=400,
                                bg=PALETTE["light_bg"]); self.canvas.pack()
        self._show_placeholder()

        cam_ctrl = tk.Frame(left, bg=PALETTE["bg"]); cam_ctrl.pack(pady=10)
        self.btn_cam = tk.Button(cam_ctrl, text="🎥 Iniciar Cámara",
                                 command=self.toggle_camera,
                                 bg=PALETTE["secondary"], fg="white",
                                 font=("Arial", 12, "bold"))
        self.btn_cam.pack(side="left", padx=5)
        tk.Button(cam_ctrl, text="🔍 Reconocer Letra",
                  command=self.realizar_prediccion,
                  bg=PALETTE["warning"], fg="white",
                  font=("Arial", 12, "bold")).pack(side="left", padx=5)

        # panel derecho
        right = tk.Frame(content, width=360, bg=PALETTE["bg"])
        right.pack(side="left", fill="y"); right.pack_propagate(False)

        lf = tk.LabelFrame(right, text="Letra Reconocida",
                           bg=PALETTE["bg"], font=("Arial", 14, "bold"))
        lf.pack(fill="x", pady=10)
        self.lbl_letra = tk.Label(lf, text="-", width=4, height=1,
                                  font=("Arial", 72, "bold"),
                                  bg=PALETTE["light_bg"],
                                  fg=PALETTE["primary"], relief="solid", bd=2)
        self.lbl_letra.pack(pady=15)

        wf = tk.LabelFrame(right, text="Palabra Formada",
                           bg=PALETTE["bg"], font=("Arial", 14, "bold"))
        wf.pack(fill="x", pady=10)
        self.lbl_palabra = tk.Label(wf, text="", height=2,
                                    font=("Arial", 18, "bold"),
                                    bg=PALETTE["light_bg"],
                                    fg=PALETTE["gray"],
                                    relief="solid", bd=1, wraplength=300,
                                    justify="center")
        self.lbl_palabra.pack(fill="x", padx=10, pady=10)

        wc = tk.Frame(right, bg=PALETTE["bg"]); wc.pack(fill="x", pady=5)
        tk.Button(wc, text="⌫ Borrar Letra",
                  command=self.borrar_letra,
                  bg=PALETTE["primary"], fg="white",
                  font=("Arial", 11, "bold")).pack(fill="x", pady=2)
        tk.Button(wc, text="🗑️ Borrar Todo",
                  command=self.borrar_todo,
                  bg=PALETTE["danger"], fg="white",
                  font=("Arial", 11, "bold")).pack(fill="x", pady=2)

        mf = tk.LabelFrame(right, text="Modos de Práctica",
                           bg=PALETTE["bg"], font=("Arial", 14, "bold"))
        mf.pack(fill="x", pady=15)
        for txt, cmd, color in (
            ("🎯 Modo Entrenamiento", self.abrir_entrenamiento, PALETTE["warning"]),
            ("🎮 Modo Juego",         self.abrir_juego,         PALETTE["secondary"]),
            ("🔤 Modo Ahorcado",      self.abrir_ahorcado,      PALETTE["primary"]),
        ):
            tk.Button(mf, text=txt, command=cmd,
                      bg=color, fg="white",
                      font=("Arial", 12, "bold")).pack(fill="x", padx=10, pady=5)

        self.status = tk.Label(self.root,
                               text="✅ Sistema listo – Selecciona un modo o inicia la cámara",
                               anchor="w", bg=PALETTE["light_bg"],
                               relief="sunken")
        self.status.pack(fill="x")

    # ---------- helpers ----------
    def _show_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 540, 400,
                                     fill=PALETTE["light_bg"], outline="")
        self.canvas.create_text(270, 200,
                                text=("📷 Cámara lista\n\n"
                                      "Haz clic en 'Iniciar Cámara'"),
                                font=("Arial", 14), fill=PALETTE["gray"],
                                justify="center")

    # ---------- cámara ----------
    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.btn_cam.config(text="⏹️ Detener Cámara")
            self.status.config(text="🎥 Reconocimiento activo")
            self.captura = cv2.VideoCapture(0)
            threading.Thread(target=self._actualizar_frame, daemon=True).start()
        else:
            self.running = False
            self.btn_cam.config(text="🎥 Iniciar Cámara")
            if self.captura: self.captura.release()
            self._show_placeholder(); self.status.config(text="⏸️ Cámara detenida")

    def _actualizar_frame(self):
        while self.running:
            ok, frame = self.captura.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = cv.hands.process(rgb)
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    cv.mp_drawing.draw_landmarks(frame, lm,
                                                 cv.mp_hands.HAND_CONNECTIONS)
            imgtk = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((540, 400)))
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk
            self.frame_actual = frame.copy()

    # ---------- reconocimiento ----------
    def realizar_prediccion(self):
        if self.frame_actual is None:
            from tkinter import messagebox
            messagebox.showwarning("Sin cámara",
                                   "Primero inicia la cámara.")
            return
        rgb = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
        res = cv.hands.process(rgb)
        if res.multi_hand_landmarks:
            coords = [coord for lm in res.multi_hand_landmarks[0].landmark
                      for coord in (lm.x, lm.y)]
            if len(coords) == 42:
                pred = cv.model.predict(np.array(coords).reshape(1, -1))[0]
                letter = cv.label_encoder.inverse_transform([pred])[0].upper()
                self.letra_actual = letter; self.lbl_letra.config(text=letter)
                self.palabra_formada += letter
                self.lbl_palabra.config(text=self.palabra_formada)
                self.status.config(text=f"✅ Letra '{letter}' añadida")

    def borrar_letra(self):
        if self.palabra_formada:
            self.palabra_formada = self.palabra_formada[:-1]
            self.lbl_palabra.config(text=self.palabra_formada)
            self.status.config(text="⌫ Última letra borrada")

    def borrar_todo(self):
        self.palabra_formada = ""; self.lbl_palabra.config(text="")
        self.lbl_letra.config(text="-"); self.letra_actual = "-"
        self.status.config(text="🗑️ Borrado completo")

    # ---------- modos ----------
    def abrir_entrenamiento(self):
        self.root.withdraw(); TrainingModeWindow(self)

    def abrir_juego(self):
        self.root.withdraw(); GameModeWindow(self)

    def abrir_ahorcado(self):
        self.root.withdraw(); HangmanModeWindow(self)

def main():
    def launch():
        root = tk.Tk(); LSPApp(root); root.mainloop()
    LoadingWindow(launch).show()


if __name__ == "__main__":
    main()
