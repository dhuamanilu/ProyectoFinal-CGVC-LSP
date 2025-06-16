# Estructura esperada:
# dataset/
# ├── A/
# │   ├── A1.jpg
# │   ├── A2.jpg
# ├── B/
# │   ├── B1.jpg
# ... hasta la letra Y (excluyendo LL, Ñ, etc.)

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt

# ========== 1. Cargar y preprocesar dataset ==========
def load_dataset(data_path, img_size=(128, 128)):
    X, y = [], []
    for label in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, label)
        if not os.path.isdir(folder_path):
            continue
        for img_file in os.listdir(folder_path):
            if not img_file.endswith(".jpg"):
                continue
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# ========== 2. Extraer características HOG ==========
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

# ========== 3. Entrenar modelo SVM ==========
def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return clf, le

# ========== 4. Guardar modelo entrenado ==========
def save_model(clf, le, model_path='modelo_letras.pkl', labels_path='labels_encoder.pkl'):
    joblib.dump(clf, model_path)
    joblib.dump(le, labels_path)
    print(f"Modelo y etiquetas guardados en {model_path} y {labels_path}")

# ========== 5. Ejecutar entrenamiento ==========
if __name__ == "__main__":
    data_dir = "dataset"  # Asegúrate de que esta carpeta esté en el mismo nivel que este script
    print("Cargando dataset...")
    X, y = load_dataset(data_dir)
    print(f"Total de imágenes: {len(X)}")

    print("Extrayendo características HOG...")
    X_hog = extract_hog_features(X)

    print("Entrenando modelo...")
    model, label_encoder = train_model(X_hog, y)

    print("Guardando modelo...")
    save_model(model, label_encoder)

    print("Proceso completo.")
