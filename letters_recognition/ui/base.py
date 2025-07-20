# lsp_app/ui/base.py
"""
Ventana base con utilidades de cámara compartidas.
Todas las subclases acceden a cv.hands / cv.model dinámicamente,
evitando copias obsoletas.
"""
import threading
import cv2
import tkinter as tk
from PIL import Image, ImageTk

from letters_recognition.common import PALETTE
import letters_recognition.common.vision as cv


class BaseCameraWindow(tk.Toplevel):
    CAM_W = 420
    CAM_H = 320

    def __init__(self, parent, title: str):
        super().__init__(parent.root)
        self.parent = parent

        self.running = False
        self.cap = None
        self.frame_actual = None

        self.title(title)
        self.configure(bg=PALETTE["bg"])
        self.geometry("900x680")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close)

        # centrar
        self.update_idletasks()
        x = (self.winfo_screenwidth() - self.winfo_reqwidth()) // 2
        y = (self.winfo_screenheight() - self.winfo_reqheight()) // 4
        self.geometry(f"+{x}+{y}")

    # -------- cámara --------
    def _start_camera(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self._update_frame, daemon=True).start()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_actual = None

    # debe implementarla cada subclase
    def _update_frame(self):
        raise NotImplementedError("Subclase debe implementar _update_frame.")

    # -------- lifecycle --------
    def close(self):
        """Detiene la cámara y destruye la ventana de forma segura."""
        self._stop_camera()

        # Espera breve para asegurar que el hilo termine antes de destruir
        self.after(100, self._safe_destroy)

    def _safe_destroy(self):
        """Intenta destruir la ventana, ignorando errores si ya fue cerrada."""
        try:
            self.destroy()
        except tk.TclError:
            pass

        # Mostrar ventana principal si es necesario
        if self.parent and hasattr(self.parent, "root"):
            self.parent.root.deiconify()
