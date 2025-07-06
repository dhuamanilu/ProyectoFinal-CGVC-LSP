# main.py
import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def launch_module(module_path):
    subprocess.Popen([sys.executable, "-m", module_path], cwd=BASE_DIR)

def open_letters():
    launch_module("letters_recognition.app_letras")

def open_words():
    launch_module("words_recognition.scripts.test_word_interface")


# Configuración de la ventana 
root = tk.Tk()
root.title("🔠 Reconocimiento de Letras y Palabras")
root.geometry("480x320")
root.resizable(False, False)

# Colores
COLOR_BG       = "#F5F7FA"   # fondo general
COLOR_FRAME    = "#FFFFFF"   # fondo de los paneles
ACCENT_COLOR   = "#4A90E2"   # color principal (azul)
ACCENT_HOVER   = "#357ABD"   # color botón al pasar el ratón
TEXT_COLOR     = "#333333"   # texto normal
SECONDARY_TEXT = "#555555"   # texto secundario

# Estilos 
style = ttk.Style(root)
style.theme_use("clam")

style.configure("TFrame",
                background=COLOR_BG)
style.configure("Card.TFrame",
                background=COLOR_FRAME,
                relief="raised",
                borderwidth=1)
style.configure("Header.TLabel",
                background=COLOR_BG,
                foreground=ACCENT_COLOR,
                font=("Segoe UI", 20, "bold"))
style.configure("TLabel",
                background=COLOR_BG,
                foreground=TEXT_COLOR,
                font=("Segoe UI", 10))
style.configure("Desc.TLabel",
                background=COLOR_FRAME,
                foreground=SECONDARY_TEXT,
                font=("Segoe UI",  nine:=9))
style.configure("Footer.TLabel",
                background=COLOR_BG,
                foreground=SECONDARY_TEXT,
                font=("Segoe UI", 9))
style.configure("Accent.TButton",
                background=ACCENT_COLOR,
                foreground="#FFFFFF",
                font=("Segoe UI", 14),
                padding=(10, 5))
style.map("Accent.TButton",
          background=[("active", ACCENT_HOVER)],
          foreground=[("active", "#FFFFFF")])


root.configure(background=COLOR_BG)
main_frame = ttk.Frame(root, style="TFrame", padding=20)
main_frame.pack(expand=True, fill="both")

# Tarjeta central
card = ttk.Frame(main_frame, style="Card.TFrame", padding=30)
card.place(relx=0.5, rely=0.5, anchor="center")

# Título
header = ttk.Label(card, text="Selecciona un Modo", style="Header.TLabel")
header.pack(pady=(0, 20))

# Botones y descripciones
btn_frame = ttk.Frame(card, style="Card.TFrame")
btn_frame.pack()

# Botón LETRA
btn_letters = ttk.Button(btn_frame, text="📖 Modo LETRA",
                         style="Accent.TButton", command=open_letters)
btn_letters.grid(row=0, column=0, padx=15, pady=(0, 5))
desc_letters = ttk.Label(btn_frame,
    text="Analiza y reconoce letras individuales",
    style="Desc.TLabel", justify="center", wraplength=140)
desc_letters.grid(row=1, column=0, padx=15)

# Botón PALABRA
btn_words = ttk.Button(btn_frame, text="🔤 Modo PALABRA",
                       style="Accent.TButton", command=open_words)
btn_words.grid(row=0, column=1, padx=15, pady=(0, 5))
desc_words = ttk.Label(btn_frame,
    text="Detecta y reconoce palabras enteras\nen la entrada de video",
    style="Desc.TLabel", justify="center", wraplength=140)
desc_words.grid(row=1, column=1, padx=15)

# Separador
sep = ttk.Separator(card, orient="horizontal")
sep.pack(fill="x", pady=20)

# Pie de página
footer = ttk.Label(card,
    text="Se abrirá una ventana independiente para cada modo",
    style="Footer.TLabel")
footer.pack()

root.mainloop()
