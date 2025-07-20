import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import numpy as np
import joblib
import os
import random
import time

from letters_recognition.config import MODEL_PATH, ENCODER_PATH, DATASET_DIR

# Variables globales para modelo y hands
model = None
label_encoder = None
mp_hands = None
mp_drawing = None
hands = None

# Función para cargar modelo en hilo separado
def load_model_and_hands():
    global model, label_encoder, mp_hands, mp_drawing, hands
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

class LoadingWindow:
    def __init__(self, parent_callback):
        self.parent_callback = parent_callback
        self.root = tk.Tk()
        self.root.title("LSP - Cargando...")
        self.root.geometry("400x200")
        self.root.resizable(False, False)
        self.root.configure(bg="#4285f4")
        
        # Centrar la ventana
        self.root.geometry("+{}+{}".format(
            int(self.root.winfo_screenwidth()/2 - 200),
            int(self.root.winfo_screenheight()/2 - 100)
        ))
        
        self._build_ui()
        self._start_loading()
        
    def _build_ui(self):
        main_frame = tk.Frame(self.root, bg="#4285f4")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(main_frame, text="LSP - Lengua de Señas Peruana", 
                font=("Arial", 16, "bold"), bg="#4285f4", fg="white").pack(pady=10)
        
        tk.Label(main_frame, text="Cargando cámara y modelo...", 
                font=("Arial", 12), bg="#4285f4", fg="white").pack(pady=5)
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=300)
        self.progress.pack(pady=20)
        self.progress.start(10)
        
        self.status_label = tk.Label(main_frame, text="Inicializando...", 
                                   font=("Arial", 10), bg="#4285f4", fg="white")
        self.status_label.pack()
        
    def _start_loading(self):
        def loading_thread():
            self.status_label.config(text="Cargando modelo de IA...")
            self.root.update()
            time.sleep(0.5)
            
            load_model_and_hands()
            
            self.status_label.config(text="Inicializando cámara...")
            self.root.update()
            time.sleep(0.5)
            
            self.status_label.config(text="¡Listo!")
            self.root.update()
            time.sleep(0.5)
            
            self.root.after(100, self._finish_loading)
            
        threading.Thread(target=loading_thread, daemon=True).start()
        
    def _finish_loading(self):
        self.progress.stop()
        self.root.destroy()
        self.parent_callback()
        
    def show(self):
        self.root.mainloop()

class GameModeWindow:
    CAM_WIDTH = 420
    CAM_HEIGHT = 320
    PALETTE = {
        "primary": "#4285f4", 
        "secondary": "#34a853",
        "danger": "#ea4335", 
        "warning": "#ff9f0a", 
        "gray": "#6c757d", 
        "bg": "#ffffff", 
        "light_bg": "#f8f9fa",
        "border": "#e9ecef"
    }
    
    def __init__(self, parent):
        self.parent = parent
        self.target_name = ""
        self.current_letter_idx = 0
        self.running = False
        self.recognized_letters = []
        self.examples = []
        self.example_idx = 0
        
        self.top = tk.Toplevel(parent.root)
        self.top.title("Modo Juego - Deletrea tu Nombre")
        self.top.geometry("900x680")
        self.top.resizable(False, False)
        self.top.configure(bg=self.PALETTE["bg"])
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
        style.configure("Game.TButton", font=("Arial", 12, "bold"))
        
    def _build_ui(self):
        # Header
        header = tk.Frame(self.top, bg=self.PALETTE["primary"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🎮 Modo Juego - Deletrea tu Nombre",
            font=("Arial", 18, "bold"),
            bg=self.PALETTE["primary"],
            fg="white"
        ).pack(pady=15)

        # Contenido principal
        content = tk.Frame(self.top, bg=self.PALETTE["bg"])
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # Frame superior para entrada de nombre
        name_frame = tk.Frame(content, bg=self.PALETTE["light_bg"], relief="solid", bd=1)
        name_frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            name_frame,
            text="Ingresa tu nombre:",
            font=("Arial", 12, "bold"),
            bg=self.PALETTE["light_bg"]
        ).pack(pady=5)

        entry_frame = tk.Frame(name_frame, bg=self.PALETTE["light_bg"])
        entry_frame.pack(pady=5)

        self.name_entry = tk.Entry(entry_frame, font=("Arial", 14), width=20, justify="center")
        self.name_entry.pack(side="left", padx=5)

        tk.Button(
            entry_frame,
            text="Empezar Juego",
            bg=self.PALETTE["secondary"],
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.start_game
        ).pack(side="left", padx=5)

        # Frame principal del juego
        game_frame = tk.Frame(content, bg=self.PALETTE["bg"])
        game_frame.pack(fill="both", expand=True)

    # Left side - Cámara
        left_frame = tk.Frame(game_frame, bg=self.PALETTE["bg"])
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cam_frame = tk.Frame(
            left_frame,
            width=self.CAM_WIDTH,
            height=self.CAM_HEIGHT,
            bg=self.PALETTE["border"],
            relief="solid",
            bd=2
        )
        cam_frame.pack(pady=5)
        cam_frame.pack_propagate(False)

        self.canvas = tk.Canvas(
            cam_frame,
            width=self.CAM_WIDTH,
            height=self.CAM_HEIGHT,
            bg=self.PALETTE["light_bg"],
            bd=0,
            highlightthickness=0
        )
        self.canvas.pack()
        self._show_loading_camera()

        # Controles de cámara
        cam_controls = tk.Frame(left_frame, bg=self.PALETTE["bg"])
        cam_controls.pack(pady=10)

        self.btn_camera = tk.Button(
            cam_controls,
            text="Iniciar Cámara",
            bg=self.PALETTE["primary"],
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.toggle_camera
        )
        self.btn_camera.pack(side="left", padx=5)

        tk.Button(
            cam_controls,
            text="Reconocer Letra",
            bg=self.PALETTE["warning"],
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.recognize_letter
        ).pack(side="left", padx=5)

        # Right side - Info del juego
        right_frame = tk.Frame(game_frame, width=350, bg=self.PALETTE["bg"])
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        # Progreso del juego
        progress_frame = tk.LabelFrame(
            right_frame,
            text="Progreso del Juego",
            font=("Arial", 12, "bold"),
            bg=self.PALETTE["bg"]
        )
        progress_frame.pack(fill="x", pady=5)

        self.name_display = tk.Label(
            progress_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg=self.PALETTE["light_bg"],
            relief="solid",
            bd=1,
            height=2
        )
        self.name_display.pack(fill="x", padx=10, pady=5)

        # Letra actual
        current_frame = tk.LabelFrame(
            right_frame,
            text="Letra Actual",
            font=("Arial", 12, "bold"),
            bg=self.PALETTE["bg"]
        )
        current_frame.pack(fill="x", pady=5)

        self.current_letter_label = tk.Label(
            current_frame,
            text="-",
            font=("Arial", 48, "bold"),
            bg=self.PALETTE["light_bg"],
            fg=self.PALETTE["primary"],
            width=3,
            height=1,
            relief="solid",
            bd=1
        )
        self.current_letter_label.pack(pady=10)

        # Ejemplos
        example_frame = tk.LabelFrame(
            right_frame,
            text="Ejemplo de la Seña",
            font=("Arial", 12, "bold"),
            bg=self.PALETTE["bg"]
        )
        example_frame.pack(fill="x", pady=5)

        self.example_pic = tk.Label(
            example_frame,
            bg=self.PALETTE["light_bg"],
            relief="solid",
            bd=1
        )
        self.example_pic.pack(padx=10, pady=5)

        nav_frame = tk.Frame(example_frame, bg=self.PALETTE["bg"])
        nav_frame.pack(pady=5)

        bottom_controls = tk.Frame(content, bg=self.PALETTE["bg"])
        bottom_controls.pack(fill="x", pady=(10, 0))

        tk.Button(
            bottom_controls,
            text="Volver",
            bg=self.PALETTE["danger"],
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.close
        ).pack(side="left", expand=True, fill="x", padx=5)

        tk.Button(
            bottom_controls,
            text="Reiniciar Juego",
            bg=self.PALETTE["warning"],
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.restart_game
        ).pack(side="left", expand=True, fill="x", padx=5)

        tk.Button(
            bottom_controls,
            text="Siguiente Letra",
            bg=self.PALETTE["secondary"],
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.next_letter
        ).pack(side="left", expand=True, fill="x", padx=5)


        # Status bar
        self.status = tk.Label(
            self.top,
            text="Ingresa tu nombre para comenzar",
            anchor="w",
            bg=self.PALETTE["light_bg"],
            relief="sunken"
        )
        self.status.pack(fill="x")

        
    def _show_loading_camera(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.CAM_WIDTH, self.CAM_HEIGHT, 
                                   fill=self.PALETTE["light_bg"], outline="")
        self.canvas.create_text(self.CAM_WIDTH // 2, self.CAM_HEIGHT // 2, 
                              text="📷 Cámara lista\nHaz clic en 'Iniciar Cámara'", 
                              font=("Arial", 12), fill=self.PALETTE["gray"], justify="center")
        
    def start_game(self):
        name = self.name_entry.get().strip().upper()
        if not name or not name.isalpha():
            messagebox.showwarning("Nombre inválido", "Por favor ingresa un nombre válido (solo letras).")
            return
            
        self.target_name = name
        self.current_letter_idx = 0
        self.recognized_letters = ["_"] * len(name)
        self.update_display()
        self.load_current_letter_examples()
        self.status.config(text=f"¡Juego iniciado! Deletrea: {name}")
        
    def update_display(self):
        if not self.target_name:
            return
            
        display_text = ""
        for i, letter in enumerate(self.target_name):
            if i == self.current_letter_idx:
                display_text += f"[{self.recognized_letters[i]}] "
            else:
                display_text += f"{self.recognized_letters[i]} "
                
        self.name_display.config(text=display_text.strip())
        
        if self.current_letter_idx < len(self.target_name):
            current = self.target_name[self.current_letter_idx]
            self.current_letter_label.config(text=current)
        else:
            self.current_letter_label.config(text="✓")
            
    def load_current_letter_examples(self):
        if not self.target_name or self.current_letter_idx >= len(self.target_name):
            return
            
        letter = self.target_name[self.current_letter_idx].lower()
        letter_path = os.path.join(DATASET_DIR, letter)
        
        if os.path.exists(letter_path):
            self.examples = [os.path.join(letter_path, f) for f in os.listdir(letter_path) 
                           if f.lower().endswith((".jpg", ".png"))]
            random.shuffle(self.examples)
            self.example_idx = 0
            self._show_example()
        else:
            self.example_pic.config(image="", text=f"No hay ejemplos\npara '{letter.upper()}'")
            
    def _show_example(self):
        if not self.examples:
            return
            
        try:
            path = self.examples[self.example_idx]
            img = Image.open(path).resize((150, 150))
            self.example_imgtk = ImageTk.PhotoImage(img)
            self.example_pic.config(image=self.example_imgtk, text="")
        except Exception:
            self.example_pic.config(image="", text="Error cargando\nejemplo")
            
    def next_example(self):
        if self.examples:
            self.example_idx = (self.example_idx + 1) % len(self.examples)
            self._show_example()
            
    def prev_example(self):
        if self.examples:
            self.example_idx = (self.example_idx - 1) % len(self.examples)
            self._show_example()
            
    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.btn_camera.config(text="Detener Cámara")
            self.canvas.delete("all")
            self.canvas.create_text(self.CAM_WIDTH // 2, self.CAM_HEIGHT // 2, 
                                text="🔄 Cargando cámara...", font=("Arial", 12), fill=self.PALETTE["gray"], justify="center")
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self._update_frame, daemon=True).start()
        else:
            self.running = False
            self.btn_camera.config(text="Iniciar Cámara")
            self._stop_camera()
            
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
                    
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((self.CAM_WIDTH, self.CAM_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk
            self.frame_actual = frame.copy()
            
        self._stop_camera()
        
    def _stop_camera(self):
        if self.cap:
            self.cap.release()
        self._show_loading_camera()
        
    def recognize_letter(self):
        if not self.target_name or self.frame_actual is None:
            return
            
        frame = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                if len(landmarks) == 42:
                    X = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(X)[0]
                    letter = label_encoder.inverse_transform([pred])[0].upper()
                    
                    target_letter = self.target_name[self.current_letter_idx]
                    if letter == target_letter:
                        self.recognized_letters[self.current_letter_idx] = letter
                        self.update_display()
                        messagebox.showinfo("¡Correcto!", f"¡Excelente! Reconociste la letra '{letter}'")
                        
                        if self.current_letter_idx < len(self.target_name) - 1:
                            self.current_letter_idx += 1
                            self.load_current_letter_examples()
                            self.update_display()
                        else:
                            messagebox.showinfo("¡Felicidades!", f"¡Completaste tu nombre: {self.target_name}!")
                            self.status.config(text="¡Juego completado!")
                    else:
                        messagebox.showinfo("Intenta de nuevo", f"Reconocí '{letter}', pero necesitas '{target_letter}'")
                        
    def next_letter(self):
        if self.target_name and self.current_letter_idx < len(self.target_name) - 1:
            self.current_letter_idx += 1
            self.load_current_letter_examples()
            self.update_display()
            
    def restart_game(self):
        if self.target_name:
            self.current_letter_idx = 0
            self.recognized_letters = ["_"] * len(self.target_name)
            self.update_display()
            self.load_current_letter_examples()
            self.status.config(text=f"Juego reiniciado. Deletrea: {self.target_name}")
            
    def close(self):
        self.running = False
        self._stop_camera()
        self.top.destroy()
        self.parent.root.deiconify()

class HangmanModeWindow:
    CAM_W, CAM_H = 420, 320
    MAX_ERRORS = 10
    PALETTE = {
        "primary": "#4285f4", "secondary": "#34a853",
        "danger": "#ea4335", "warning": "#ff9f0a",
        "gray": "#6c757d", "bg": "#ffffff",
        "light_bg": "#f8f9fa", "border": "#e9ecef"
    }
    WORD_LIST = ["B","PERRO", "GATO", "CASA", "AMIGO", "FAMILIA", "PYTHON", "AHORCADO"]

    def __init__(self, parent):
        self.parent = parent
        self.errors = 0
        self.guessed = set()
        self.word = ""
        self.frame_actual = None
        self.running = False

        self.top = tk.Toplevel(parent.root)
        self.top.title("Modo Ahorcado - LSP")
        self.top.geometry("900x680")
        self.top.resizable(False, False)
        self.top.configure(bg=self.PALETTE["bg"])
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()
        self.reset_game()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.top, bg=self.PALETTE["primary"], height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🔤 Modo Ahorcado",
                 font=("Arial", 18, "bold"), bg=self.PALETTE["primary"],
                 fg="white").pack(pady=15)

        # Main content
        content = tk.Frame(self.top, bg=self.PALETTE["bg"])
        content.pack(expand=True, fill="both", padx=15, pady=15)

        # Left: cámara y control
        left = tk.Frame(content, bg=self.PALETTE["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0,10))
        cam_fr = tk.Frame(left, width=self.CAM_W, height=self.CAM_H,
                          bg=self.PALETTE["border"], bd=2, relief="solid")
        cam_fr.pack(pady=5)
        cam_fr.pack_propagate(False)
        self.canvas = tk.Canvas(cam_fr, width=self.CAM_W, height=self.CAM_H,
                                bg=self.PALETTE["light_bg"])
        self.canvas.pack()
        self._show_cam_placeholder()
        btns = tk.Frame(left, bg=self.PALETTE["bg"])
        btns.pack(pady=10)
        self.btn_cam = tk.Button(btns, text="Iniciar Cámara",
                                 bg=self.PALETTE["primary"], fg="white",
                                 font=("Arial",11,"bold"),
                                 command=self.toggle_camera)
        self.btn_cam.pack(side="left", padx=5)
        tk.Button(btns, text="Reconocer Letra",
                  bg=self.PALETTE["warning"], fg="white",
                  font=("Arial",11,"bold"),
                  command=self.recognize_letter).pack(side="left", padx=5)

        # Right: estado del juego
        right = tk.Frame(content, width=300, bg=self.PALETTE["bg"])
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # Palabra y guiones
        pf = tk.LabelFrame(right, text="Palabra", font=("Arial",12,"bold"),
                           bg=self.PALETTE["bg"])
        pf.pack(fill="x", pady=5)
        self.lbl_word = tk.Label(pf, text="", font=("Arial",24,"bold"),
                                 bg=self.PALETTE["light_bg"], width=15, height=2,
                                 relief="solid", bd=1)
        self.lbl_word.pack(padx=10,pady=10)

        # Letras intentadas
        lf = tk.LabelFrame(right, text="Letras Intentadas",
                           font=("Arial",12,"bold"), bg=self.PALETTE["bg"])
        lf.pack(fill="x", pady=5)
        self.lbl_tried = tk.Label(lf, text="", font=("Arial",14),
                                  bg=self.PALETTE["light_bg"], height=2,
                                  wraplength=280, justify="left")
        self.lbl_tried.pack(padx=10,pady=10)

        # Ahorcado (canvas de dibujo)
        hf = tk.LabelFrame(right, text="Ahorcado",
                           font=("Arial",12,"bold"), bg=self.PALETTE["bg"])
        hf.pack(fill="both", expand=True, pady=5)
        self.hang_canvas = tk.Canvas(hf, width=200, height=240,
                                     bg=self.PALETTE["light_bg"])
        self.hang_canvas.pack(padx=10,pady=10)

        # Controles inferiores
        ctrls = tk.Frame(self.top, bg=self.PALETTE["bg"])
        ctrls.pack(fill="x", pady=(0,10))
        tk.Button(ctrls, text="Reiniciar", bg=self.PALETTE["secondary"],
                  fg="white", font=("Arial",11,"bold"),
                  command=self.reset_game).pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(ctrls, text="Volver", bg=self.PALETTE["danger"],
                  fg="white", font=("Arial",11,"bold"),
                  command=self.close).pack(side="left", expand=True, fill="x", padx=5)

    def _show_cam_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0,0,self.CAM_W,self.CAM_H,
                                     fill=self.PALETTE["light_bg"], outline="")
        self.canvas.create_text(self.CAM_W//2, self.CAM_H//2,
                                text="📷 Cámara lista\nHaz clic en 'Iniciar Cámara'",
                                font=("Arial",12), fill=self.PALETTE["gray"],
                                justify="center")

    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.btn_cam.config(text="Detener Cámara")
            self.canvas.delete("all")
            self.canvas.create_text(self.CAM_W//2,self.CAM_H//2,
                                    text="🔄 Cargando cámara...",
                                    font=("Arial",12),fill=self.PALETTE["gray"])
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self._update_frame, daemon=True).start()
        else:
            self.running = False
            self.btn_cam.config(text="Iniciar Cámara")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            self._show_cam_placeholder()

    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((self.CAM_W,self.CAM_H))
            imgtk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0,0,anchor="nw",image=imgtk)
            self.canvas.imgtk = imgtk
            self.frame_actual = frame.copy()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self._show_cam_placeholder()

    def reset_game(self):
        self.errors = 0
        self.guessed.clear()
        self.word = random.choice(self.WORD_LIST)
        self._update_word_display()
        self._update_tried()
        self._draw_hangman()

    def _update_word_display(self):
        disp = " ".join([c if c in self.guessed else "_" for c in self.word])
        self.lbl_word.config(text=disp)

    def _update_tried(self):
        self.lbl_tried.config(text=" ".join(sorted(self.guessed)))

    def _draw_hangman(self):
        self.hang_canvas.delete("all")
        parts = [
            lambda c: c.create_line(20,220,180,220),        # suelo
            lambda c: c.create_line(50,220,50,20),          # poste vertical
            lambda c: c.create_line(50,20,140,20),          # viga horizontal
            lambda c: c.create_line(140,20,140,50),         # cuerda
            lambda c: c.create_oval(120,50,160,90),         # cabeza
            lambda c: c.create_line(140,90,140,150),        # cuerpo
            lambda c: c.create_line(140,110,120,130),       # brazo izq
            lambda c: c.create_line(140,110,160,130),       # brazo der
            lambda c: c.create_line(140,150,120,180),       # pierna izq
            lambda c: c.create_line(140,150,160,180),       # pierna der
        ]
        # dibujar según número de errores (hasta MAX_ERRORS)
        for i in range(min(self.errors, self.MAX_ERRORS)):
            parts[i](self.hang_canvas)

    def recognize_letter(self):
        if not hasattr(self, "frame_actual") or self.frame_actual is None:
            return
        rgb = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            messagebox.showinfo("Sin mano", "No se detectó mano. Intenta de nuevo.")
            return
        for lm in results.multi_hand_landmarks:
            coords = [coord for lm in results.multi_hand_landmarks[0].landmark for coord in (lm.x, lm.y)]
            if len(coords) != 42:
                continue
            X = np.array(coords).reshape(1, -1)
            pred = model.predict(X)[0]
            letter = label_encoder.inverse_transform([pred])[0].upper()
            if letter in self.guessed:
                messagebox.showinfo("Letra repetida", f"Ya probaste '{letter}'.")
                return
            self.guessed.add(letter)
            if letter in self.word:
                self._update_word_display()
                if all(c in self.guessed for c in self.word):
                    messagebox.showinfo("¡Ganaste!", f"¡Felicidades! La palabra era '{self.word}'.")
            else:
                self.errors += 1
                self._draw_hangman()
                if self.errors >= self.MAX_ERRORS:
                    messagebox.showinfo("Perdiste", f"Se acabó el juego. La palabra era '{self.word}'.")
            self._update_tried()
            return

    def close(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.top.destroy()
        self.parent.root.deiconify()

class TrainingModeWindow:
    CAM_WIDTH = 420
    CAM_HEIGHT = 320
    PALETTE = {
        "primary": "#4285f4", 
        "secondary": "#34a853",
        "danger": "#ea4335", 
        "warning": "#ff9f0a", 
        "gray": "#6c757d", 
        "bg": "#ffffff", 
        "light_bg": "#f8f9fa",
        "border": "#e9ecef"
    }
    
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
        self.top.geometry("900x680")
        self.top.resizable(False, False)
        self.top.configure(bg=self.PALETTE["bg"])
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
        style.configure("Red.Horizontal.TProgressbar", troughcolor=self.PALETTE["border"], 
                       bordercolor=self.PALETTE["border"], background=self.PALETTE["danger"], thickness=16)
        style.configure("Yellow.Horizontal.TProgressbar", troughcolor=self.PALETTE["border"], 
                       bordercolor=self.PALETTE["border"], background=self.PALETTE["warning"], thickness=16)
        style.configure("Green.Horizontal.TProgressbar", troughcolor=self.PALETTE["border"], 
                       bordercolor=self.PALETTE["border"], background=self.PALETTE["secondary"], thickness=16)
                       
    def _build_ui(self):
        # Header
        header = tk.Frame(self.top, bg=self.PALETTE["primary"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="🎯 Modo Entrenamiento – Lengua de Señas Peruana", 
             font=("Arial", 18, "bold"), bg=self.PALETTE["primary"], fg="white").pack(pady=15)

        # Controles superiores fijos (debajo del header)
        top_controls = tk.Frame(self.top, bg=self.PALETTE["bg"])
        top_controls.pack(fill="x", pady=(5, 0))

        self.btn_practice = tk.Button(top_controls, text="Iniciar Práctica", 
                                  bg=self.PALETTE["secondary"], fg="white",
                                  font=("Arial", 11, "bold"), command=self.toggle_practice)
        self.btn_practice.pack(side="left", padx=10)

        self.btn_volver = tk.Button(top_controls, text="Volver", 
                                bg=self.PALETTE["danger"], fg="white",
                                font=("Arial", 11, "bold"), command=self.close)
        self.btn_volver.pack(side="left")

        # Contenido principal
        content = tk.Frame(self.top, bg=self.PALETTE["bg"])
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # Left side - Camera
        left_frame = tk.Frame(content, bg=self.PALETTE["bg"])
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cam_frame = tk.Frame(left_frame, width=self.CAM_WIDTH, height=self.CAM_HEIGHT, 
                         bg=self.PALETTE["border"], relief="solid", bd=2)
        cam_frame.pack(pady=5)
        cam_frame.pack_propagate(False)

        self.canvas = tk.Canvas(cam_frame, width=self.CAM_WIDTH, height=self.CAM_HEIGHT, 
                            bg=self.PALETTE["light_bg"], bd=0, highlightthickness=0)
        self.canvas.pack()
        self._show_loading_camera()

        # Right side - Training info
        right_frame = tk.Frame(content, width=350, bg=self.PALETTE["bg"])
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        # Selector de letra
        selector_frame = tk.LabelFrame(right_frame, text="Seleccionar Letra", font=("Arial", 12, "bold"),
                                   bg=self.PALETTE["bg"])
        selector_frame.pack(fill="x", pady=5)

        letras = [l.upper() for l in sorted(os.listdir(DATASET_DIR)) if len(l) == 1 and l.isalpha()]
        self.combo = ttk.Combobox(selector_frame, state="readonly", values=letras, 
                              font=("Arial", 14), width=8, justify="center")
        self.combo.pack(pady=10)
        self.combo.bind("<<ComboboxSelected>>", self._on_letter_select)

        # Objetivo actual
        target_frame = tk.LabelFrame(right_frame, text="Letra Objetivo", font=("Arial", 12, "bold"),
                                 bg=self.PALETTE["bg"])
        target_frame.pack(fill="x", pady=5)

        self.lbl_objetivo = tk.Label(target_frame, text="-", font=("Arial", 48, "bold"),
                                 bg=self.PALETTE["light_bg"], fg=self.PALETTE["primary"],
                                 width=3, height=1, relief="solid", bd=1)
        self.lbl_objetivo.pack(pady=10)

        # Progreso
        progress_frame = tk.LabelFrame(right_frame, text="Precisión", font=("Arial", 12, "bold"),
                                   bg=self.PALETTE["bg"])
        progress_frame.pack(fill="x", pady=5)

        self.precision_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(progress_frame, variable=self.precision_var, 
                                    style="Red.Horizontal.TProgressbar", maximum=100, length=280)
        self.progress.pack(padx=10, pady=5)

        self.lbl_precision = tk.Label(progress_frame, text="0% de precisión", 
                                  font=("Arial", 11), bg=self.PALETTE["bg"])
        self.lbl_precision.pack(pady=5)

    # Ejemplos
        example_frame = tk.LabelFrame(right_frame, text="Ejemplos", font=("Arial", 12, "bold"),
                                  bg=self.PALETTE["bg"])
        example_frame.pack(fill="x", pady=5)

        self.example_pic = tk.Label(example_frame, bg=self.PALETTE["light_bg"], relief="solid", bd=1)
        self.example_pic.pack(padx=10, pady=5)

        nav_frame = tk.Frame(example_frame, bg=self.PALETTE["bg"])
        nav_frame.pack(pady=5)

        tk.Button(nav_frame, text="◀", width=4, command=self.prev_example,
              bg=self.PALETTE["gray"], fg="white").pack(side="left", padx=2)
        tk.Button(nav_frame, text="▶", width=4, command=self.next_example,
              bg=self.PALETTE["gray"], fg="white").pack(side="left", padx=2)

        # Status bar
        self.status = tk.Label(self.top, text="Selecciona una letra para comenzar", anchor="w", 
                           bg=self.PALETTE["light_bg"], relief="sunken")
        self.status.pack(fill="x")

        
    def _show_loading_camera(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.CAM_WIDTH, self.CAM_HEIGHT, 
                                   fill=self.PALETTE["light_bg"], outline="")
        self.canvas.create_text(self.CAM_WIDTH // 2, self.CAM_HEIGHT // 2, 
                              text="📷 Cámara lista para entrenamiento", 
                              font=("Arial", 12), fill=self.PALETTE["gray"])
                              
    def _on_letter_select(self, *_):
        letra = self.combo.get().lower()
        if letra:
            self.set_target_letter(letra)
            
    def set_target_letter(self, letter):
        self.target_letter = letter
        self.lbl_objetivo.config(text=letter.upper())
        self.precision_var.set(0)
        self.lbl_precision.config(text="0% de precisión")
        self.correct_counter = 0
        self.total_counter = 0
        self.examples = []
        self.example_pic.config(image="", text="")
        self.status.config(text=f"Letra objetivo '{letter.upper()}' seleccionada")
        self.load_examples()
        
    def toggle_practice(self):
        if not self.target_letter:
            messagebox.showinfo("Selecciona una letra", "Debes escoger una letra antes de practicar.")
            return
            
        if not self.running:
            self.running = True
            self.btn_practice.config(text="Detener Práctica")
            self.canvas.delete("all")
            self.canvas.create_text(self.CAM_WIDTH // 2, self.CAM_HEIGHT // 2, 
                                text="🔄 Cargando cámara...", font=("Arial", 12), fill=self.PALETTE["gray"], justify="center")
            self.cap = cv2.VideoCapture(0)
            self.status.config(text="Práctica en curso...")
            threading.Thread(target=self._update_frame, daemon=True).start()
        else:
            self.running = False
            self.btn_practice.config(text="Iniciar Práctica")
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
                self.lbl_precision.config(text=f"{precision:.0f}% de precisión")
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
        self._show_loading_camera()
        self.status.config(text="Práctica detenida")
        
    def load_examples(self):
        if not self.target_letter:
            return
            
        ruta = os.path.join(DATASET_DIR, self.target_letter)
        if os.path.exists(ruta):
            self.examples = [os.path.join(ruta, f) for f in os.listdir(ruta) 
                           if f.lower().endswith((".jpg", ".png"))]
            if self.examples:
                random.shuffle(self.examples)
                self.example_idx = 0
                self._show_example()
                
    def _show_example(self):
        if not self.examples:
            return
            
        try:
            path = self.examples[self.example_idx]
            img = Image.open(path).resize((150, 150))
            self.example_imgtk = ImageTk.PhotoImage(img)
            self.example_pic.config(image=self.example_imgtk, text="")
        except Exception:
            self.example_pic.config(image="", text="Error cargando ejemplo")
            
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
    PALETTE = {
        "primary": "#4285f4", 
        "secondary": "#34a853",
        "danger": "#ea4335", 
        "warning": "#ff9f0a", 
        "gray": "#6c757d", 
        "bg": "#ffffff", 
        "light_bg": "#f8f9fa",
        "border": "#e9ecef"
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("LSP - Lengua de Señas Peruana")
        self.root.geometry("980x720")
        self.root.resizable(False, False)
        self.root.configure(bg=self.PALETTE["bg"])
        
        self.letra_actual = "-"
        self.palabra_formada = ""
        self.captura = None
        self.running = False
        self.frame_actual = None
        
        self._build_main_ui()
        
    def _build_main_ui(self):
        # Header
        header = tk.Frame(self.root, bg=self.PALETTE["primary"], height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="🤟 LSP - Lengua de Señas Peruana", 
                bg=self.PALETTE["primary"], fg="white", 
                font=("Arial", 20, "bold")).pack(pady=20)
        
        # Content
        content = tk.Frame(self.root, bg=self.PALETTE["bg"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Camera
        left_frame = tk.Frame(content, bg=self.PALETTE["bg"])
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 15))
        
        cam_frame = tk.Frame(left_frame, width=540, height=400, 
                            bg=self.PALETTE["border"], relief="solid", bd=2)
        cam_frame.pack(pady=10)
        cam_frame.pack_propagate(False)
        
        self.canvas = tk.Canvas(cam_frame, width=540, height=400, 
                               bg=self.PALETTE["light_bg"], bd=0, highlightthickness=0)
        self.canvas.pack()
        self._show_loading_camera()
        
        # Camera controls
        cam_controls = tk.Frame(left_frame, bg=self.PALETTE["bg"])
        cam_controls.pack(pady=10)
        
        self.btn_camera = tk.Button(cam_controls, text="🎥 Iniciar Cámara", 
                                   bg=self.PALETTE["secondary"], fg="white",
                                   font=("Arial", 12, "bold"), command=self.toggle_camera)
        self.btn_camera.pack(side="left", padx=5)
        
        tk.Button(cam_controls, text="🔍 Reconocer Letra", bg=self.PALETTE["warning"], fg="white",
                 font=("Arial", 12, "bold"), command=self.realizar_prediccion).pack(side="left", padx=5)
        
        # Right side - Recognition results
        right_frame = tk.Frame(content, width=360, bg=self.PALETTE["bg"])
        right_frame.pack(side=tk.LEFT, fill="y")
        right_frame.pack_propagate(False)
        
        # Recognized letter
        letter_frame = tk.LabelFrame(right_frame, text="Letra Reconocida", 
                                   font=("Arial", 14, "bold"), bg=self.PALETTE["bg"])
        letter_frame.pack(fill="x", pady=10)
        
        self.lbl_letra = tk.Label(letter_frame, text=self.letra_actual, 
                                 font=("Arial", 72, "bold"), bg=self.PALETTE["light_bg"], 
                                 fg=self.PALETTE["primary"], width=4, height=1, relief="solid", bd=2)
        self.lbl_letra.pack(pady=15)
        
        # Word formed
        word_frame = tk.LabelFrame(right_frame, text="Palabra Formada", 
                                 font=("Arial", 14, "bold"), bg=self.PALETTE["bg"])
        word_frame.pack(fill="x", pady=10)
        
        self.lbl_palabra = tk.Label(word_frame, text=self.palabra_formada, 
                                   font=("Arial", 18, "bold"), bg=self.PALETTE["light_bg"], 
                                   fg=self.PALETTE["gray"], height=2, relief="solid", bd=1, 
                                   wraplength=300, justify="center")
        self.lbl_palabra.pack(fill="x", padx=10, pady=10)
        
        # Word controls
        word_controls = tk.Frame(right_frame, bg=self.PALETTE["bg"])
        word_controls.pack(fill="x", pady=5)
        
        tk.Button(word_controls, text="⌫ Borrar Letra", bg=self.PALETTE["primary"], fg="white",
                 font=("Arial", 11, "bold"), command=self.borrar_letra).pack(fill="x", pady=2)
        
        tk.Button(word_controls, text="🗑️ Borrar Todo", bg=self.PALETTE["danger"], fg="white",
                 font=("Arial", 11, "bold"), command=self.borrar_todo).pack(fill="x", pady=2)
        
        # Mode controls
        mode_frame = tk.LabelFrame(right_frame, text="Modos de Práctica", 
                                 font=("Arial", 14, "bold"), bg=self.PALETTE["bg"])
        mode_frame.pack(fill="x", pady=15)
        
        tk.Button(mode_frame, text="🎯 Modo Entrenamiento", bg=self.PALETTE["warning"], fg="white",
                 font=("Arial", 12, "bold"), command=self.abrir_modo_entrenamiento).pack(fill="x", padx=10, pady=5)
        
        tk.Button(mode_frame, text="🎮 Modo Juego", bg=self.PALETTE["secondary"], fg="white",
                 font=("Arial", 12, "bold"), command=self.abrir_modo_juego).pack(fill="x", padx=10, pady=5)
        
        # Dentro de LSPApp._build_main_ui(), en el frame de modos:
        btn_hang = tk.Button(
            mode_frame,
            text="🔤 Modo Ahorcado",
            bg=self.PALETTE["primary"],
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.abrir_modo_ahorcado
        )
        btn_hang.pack(fill="x", padx=10, pady=5)

        
        # Status bar
        self.estado = tk.Label(self.root, text="✅ Sistema listo - Selecciona un modo o inicia la cámara", 
                              anchor=tk.W, bg=self.PALETTE["light_bg"], relief="sunken",
                              font=("Arial", 10))
        self.estado.pack(fill=tk.X)
        
    def _show_loading_camera(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 540, 400, fill=self.PALETTE["light_bg"], outline="")
        self.canvas.create_text(270, 200, text="📷 Cámara lista\n\nHaz clic en 'Iniciar Cámara'\npara comenzar el reconocimiento", 
                              font=("Arial", 14), fill=self.PALETTE["gray"], justify="center")
        
    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.btn_camera.config(text="⏹️ Detener Cámara")
            self.canvas.delete("all")  # LIMPIA EL CANVAS
            self.canvas.create_text(270, 200, text="🔄 Cargando cámara...", 
                                font=("Arial", 14), fill=self.PALETTE["gray"], justify="center")
            self.captura = cv2.VideoCapture(0)
            self.estado.config(text="🎥 Reconocimiento activo - Muestra tu mano frente a la cámara")
            threading.Thread(target=self._actualizar_frame, daemon=True).start()
        else:
            self.running = False
            self.btn_camera.config(text="🎥 Iniciar Cámara")
            if self.captura:
                self.captura.release()
            self._show_loading_camera()
            self.estado.config(text="⏸️ Cámara detenida - Lista para usar")
            
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
            img = img.resize((540, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
            self.frame_actual = frame.copy()
            
    def realizar_prediccion(self):
        if self.frame_actual is None:
            messagebox.showwarning("Sin cámara", "Primero inicia la cámara para poder reconocer letras.")
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
                    self.estado.config(text=f"✅ Letra '{self.letra_actual}' reconocida y agregada")
        else:
            messagebox.showinfo("Sin mano detectada", "No se detectó una mano en la imagen. Asegúrate de mostrar tu mano claramente.")
            
    def borrar_letra(self):
        if self.palabra_formada:
            self.palabra_formada = self.palabra_formada[:-1]
            self.lbl_palabra.config(text=self.palabra_formada)
            self.estado.config(text="⌫ Última letra borrada")
        else:
            self.estado.config(text="❌ No hay letras para borrar")
            
    def borrar_todo(self):
        self.palabra_formada = ""
        self.lbl_palabra.config(text="")
        self.lbl_letra.config(text="-")
        self.letra_actual = "-"
        self.estado.config(text="🗑️ Palabra y letra actual borradas")
        
    def abrir_modo_entrenamiento(self):
        self.root.withdraw()
        TrainingModeWindow(self)
        
    def abrir_modo_juego(self):
        self.root.withdraw()
        GameModeWindow(self)
    
    def abrir_modo_ahorcado(self):
        self.root.withdraw()
        HangmanModeWindow(self)


def main():
    def start_main_app():
        root = tk.Tk()
        app = LSPApp(root)
        root.mainloop()
        
    # Mostrar ventana de carga
    loading_window = LoadingWindow(start_main_app)
    loading_window.show()

if __name__ == "__main__":
    main()