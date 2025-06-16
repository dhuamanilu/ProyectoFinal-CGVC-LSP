# app_letras.py – Interfaz de reconocimiento en tiempo real (Modo Letra)

import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ======== Configuración ========
model_path = 'modelo_letras.pkl'
labels_path = 'labels_encoder.pkl'

# ======== Cargar modelo y codificador ========
model = joblib.load(model_path)
label_encoder = joblib.load(labels_path)

# ======== Función para extraer HOG de una imagen ========
def preprocess_and_extract(img, size=(128, 128)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, size)
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features.reshape(1, -1)

# ======== Iniciar cámara ========
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar región de interés (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    try:
        features = preprocess_and_extract(roi)
        prediction = model.predict(features)[0]
        letter = label_encoder.inverse_transform([prediction])[0]
        cv2.putText(frame, f"Letra: {letter.upper()}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, "Letra: ...", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento de Letras LSP", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
