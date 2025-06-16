# app_letras.py – Interfaz gráfica en tiempo real usando MediaPipe + modelo entrenado

import cv2
import mediapipe as mp
import numpy as np
import joblib

# ======== Cargar modelo y etiquetas ========
model = joblib.load('modelo_letras.pkl')
label_encoder = joblib.load('labels_encoder.pkl')

# ======== Inicializar MediaPipe Hands ========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5)

# ======== Iniciar cámara ========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if len(landmarks) == 42:
                X = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(X)[0]
                letter = label_encoder.inverse_transform([prediction])[0]

                cv2.putText(frame, f"Letra: {letter.upper()}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Mano no detectada", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Reconocimiento de Letras - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
