# entrenar_mediapipe.py – Entrenamiento usando puntos clave de MediaPipe como entrada

import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ======== Inicializar MediaPipe Hands ========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5)

# ======== Cargar imágenes y extraer puntos clave ========
def load_data_with_landmarks(data_dir):
    X, y = [], []
    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for img_name in os.listdir(label_path):
            if not img_name.endswith('.jpg'):
                continue
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                if len(landmarks) == 42:
                    X.append(landmarks)
                    y.append(label)
    return np.array(X), np.array(y)

# ======== Entrenamiento ========
def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return clf, le

# ======== Guardar modelo ========
def save_model(model, label_encoder, model_path='modelo_letras.pkl', label_path='labels_encoder.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_path)
    print(f"Modelo y etiquetas guardados en {model_path} y {label_path}")

# ======== Ejecutar ========
if __name__ == '__main__':
    dataset_path = 'dataset'
    print("Cargando imágenes y extrayendo landmarks con MediaPipe...")
    X, y = load_data_with_landmarks(dataset_path)
    print(f"Total de muestras válidas: {len(X)}")

    print("Entrenando modelo...")
    model, label_encoder = train_model(X, y)

    print("Guardando modelo...")
    save_model(model, label_encoder)

    hands.close()
    print("Entrenamiento completado.")
