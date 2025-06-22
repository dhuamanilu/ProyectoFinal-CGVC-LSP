# words_recognition/scripts/augment_landmarks.py

import numpy as np
import pickle
import os

def augment_sequence(sequence, num_augmentations_per_sample=5):
    """
    Genera versiones aumentadas de una secuencia de landmarks.

    Args:
        sequence (list of lists): Una secuencia de frames, donde cada frame
                                  es una lista de coordenadas [x1, y1, z1, x2, y2, z2, ...].
        num_augmentations_per_sample (int): Número de versiones aumentadas a generar.

    Returns:
        list of lists: Una lista que contiene la secuencia original y sus versiones aumentadas.
    """
    augmented_sequences = [sequence] # Incluye la secuencia original

    sequence_np = np.array(sequence) # Convierte a NumPy para operaciones más fáciles

    # Rango de variación para cada tipo de aumento
    translation_range = 0.02    # Porcentaje del ancho/alto para traslación
    scale_range = 0.05          # Porcentaje para escalado (ej. 0.95 a 1.05)
    noise_strength = 0.005      # Fuerza del ruido aleatorio

    for _ in range(num_augmentations_per_sample):
        temp_sequence = np.copy(sequence_np) # Copia para cada aumento

        # 1. Traslación Aleatoria (shifting)
        # Mueve todos los landmarks de cada frame ligeramente
        dx = np.random.uniform(-translation_range, translation_range)
        dy = np.random.uniform(-translation_range, translation_range)
        temp_sequence[:, ::3] += dx   # Ajusta todas las coordenadas X
        temp_sequence[:, 1::3] += dy  # Ajusta todas las coordenadas Y

        # 2. Escalado Aleatorio (scaling)
        # Escala los landmarks respecto al centro de la mano/frame
        scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
        
        # Calcula el centroide actual (media de X y Y de todos los landmarks)
        center_x = np.mean(temp_sequence[:, ::3]) 
        center_y = np.mean(temp_sequence[:, 1::3]) 

        temp_sequence[:, ::3] = center_x + (temp_sequence[:, ::3] - center_x) * scale_factor
        temp_sequence[:, 1::3] = center_y + (temp_sequence[:, 1::3] - center_y) * scale_factor
        
        # La coordenada Z también puede escalarse si la incluiste y tiene un significado relativo
        # Esto solo se aplica si la dimensión de las características es un múltiplo de 3 (x,y,z)
        if temp_sequence.shape[1] % 3 == 0: 
             center_z = np.mean(temp_sequence[:, 2::3])
             temp_sequence[:, 2::3] = center_z + (temp_sequence[:, 2::3] - center_z) * scale_factor


        # 3. Ruido Gaussiano (adding noise)
        # Añade un pequeño ruido aleatorio a cada coordenada
        temp_sequence += np.random.normal(loc=0.0, scale=noise_strength, size=temp_sequence.shape)

        # Opcional: Recorte/Ajuste de límites para mantener en [0,1] si los landmarks están normalizados
        temp_sequence = np.clip(temp_sequence, 0.0, 1.0) # Limita los valores entre 0 y 1

        augmented_sequences.append(temp_sequence.tolist()) # Vuelve a convertir a lista y añade

    return augmented_sequences

def apply_augmentations_to_dataset(input_data_file, output_data_file, num_augmentations_per_sample=5):
    """
    Carga el dataset de landmarks, aplica el aumento de datos y guarda el nuevo dataset.

    Args:
        input_data_file (str): Ruta al archivo .pkl de entrada con los landmarks.
        output_data_file (str): Ruta al archivo .pkl de salida para el dataset aumentado.
        num_augmentations_per_sample (int): Número de versiones aumentadas por cada muestra original.
    """
    print(f"Cargando datos desde {input_data_file}...")
    try:
        with open(input_data_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en la ruta esperada: {input_data_file}")
        print("Asegúrate de que 'extract_video_landmarks.py' se ejecutó con éxito y generó 'data_landmarks_sequences.pkl' en el directorio 'words_recognition/data/processed'.")
        exit() # Termina el script si el archivo no existe

    original_sequences = data['sequences']
    original_labels = data['labels']

    augmented_sequences = []
    augmented_labels = []

    print(f"Generando {num_augmentations_per_sample} aumentaciones por cada una de las {len(original_sequences)} secuencias originales...")

    for i, seq in enumerate(original_sequences):
        label = original_labels[i]
        
        # Genera versiones aumentadas de la secuencia actual
        current_augmented_versions = augment_sequence(seq, num_augmentations_per_sample)
        
        for aug_seq in current_augmented_versions:
            augmented_sequences.append(aug_seq)
            augmented_labels.append(label)

    print(f"Dataset aumentado creado. Total de secuencias: {len(augmented_sequences)}")

    # Guarda el dataset aumentado
    with open(output_data_file, 'wb') as f:
        pickle.dump({'sequences': augmented_sequences, 'labels': augmented_labels}, f)
    print(f"Dataset aumentado guardado en {output_data_file}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_data_dir = os.path.join(script_dir, '..', 'data', 'processed')
    input_pkl_file = os.path.join(input_data_dir, 'data_landmarks_sequences.pkl')

    output_data_dir = os.path.join(script_dir, '..', 'data', 'augmented')
    output_pkl_file = os.path.join(output_data_dir, 'data_landmarks_sequences_augmented.pkl')
    
    os.makedirs(output_data_dir, exist_ok=True) 

    num_augment = 5 
    
    apply_augmentations_to_dataset(input_pkl_file, output_pkl_file, num_augment)
    print("Proceso de aumento de datos completado.")