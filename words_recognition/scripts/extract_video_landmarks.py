# words_recognition/scripts/extract_video_landmarks.py

import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Inicializar MediaPipe Holistic en lugar de solo Hands
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        current_frame_features = []

        # Extraer landmarks de Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                current_frame_features.extend([lm.x, lm.y, lm.z])
        else:
            # Añadir ceros si no se detecta pose (33 landmarks * 3 coords = 99 features)
            current_frame_features.extend(np.zeros(33 * 3).tolist())

        # Extraer landmarks de Mano Izquierda
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                current_frame_features.extend([lm.x, lm.y, lm.z])
        else:
            # Añadir ceros si no se detecta mano izquierda (21 landmarks * 3 coords = 63 features)
            current_frame_features.extend(np.zeros(21 * 3).tolist())

        # Extraer landmarks de Mano Derecha
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                current_frame_features.extend([lm.x, lm.y, lm.z])
        else:
            # Añadir ceros si no se detecta mano derecha (21 landmarks * 3 coords = 63 features)
            current_frame_features.extend(np.zeros(21 * 3).tolist())
        
        # Esto asegura consistencia en la longitud de las secuencias de características.
        expected_features = (33+21+21) * 3 
        if len(current_frame_features) == expected_features:
            frame_landmarks.append(current_frame_features)
        else:
            print(f"Advertencia: El frame en {video_path} tuvo {len(current_frame_features)} características, se esperaban {expected_features}. Ignorando este frame.")


    cap.release()
    return frame_landmarks

def process_dataset_videos(dataset_dir, output_file):
    all_sequences = []
    all_labels = []

    # Crear el directorio de salida si no existe
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    SELECTED_WORDS = [
        "HOLA",
        "BIEN",
        "AMIGO",
        "CASA",
        "DINERO",
        "HACER",
        "IR",
        "LO SIENTO",
        "MUCHO",
        "UNO"
    ]
    for label in sorted(SELECTED_WORDS):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Procesando categoría: {label}")
        video_files = [f for f in os.listdir(label_path) if f.endswith('.mp4')]

        for video_name in video_files:
            # FILTRO CRUCIAL AQUÍ: Ignora videos que contengan "ORACION"
            if "ORACION" in video_name.upper():
                print(f"   Saltando video de oración: {video_name}")
                continue
            video_path = os.path.join(label_path, video_name)
            print(f"   Extrayendo de: {video_name}")
            landmarks_sequence = extract_landmarks_from_video(video_path)

            if landmarks_sequence:
                all_sequences.append(landmarks_sequence)
                all_labels.append(label)
            else:
                print(f"   Advertencia: No se detectaron landmarks válidos en {video_name}. Ignorando.")

    with open(output_file, 'wb') as f:
        pickle.dump({'sequences': all_sequences, 'labels': all_labels}, f)
    print(f"Datos de landmarks guardados en {output_file}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = os.path.join(script_dir, '..', 'data', 'raw')
    
    output_dir = os.path.join(script_dir, '..', 'data', 'processed')
    output_data_file = os.path.join(output_dir, 'data_landmarks_sequences.pkl')
    
    # Asegúrate de que el directorio de salida para los datos procesados exista
    os.makedirs(output_dir, exist_ok=True)

    process_dataset_videos(dataset_path, output_data_file)
    holistic.close() # Cerrar la instancia de MediaPipe Holistic
    print("Extracción de landmarks de videos completada.")