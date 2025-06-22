# words_recognition/scripts/train_word_model.py

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import joblib

# Desactiva las advertencias de TensorFlow sobre oneDNN (opcional, solo para limpiar la consola)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Iniciando el script de entrenamiento del modelo de palabras...")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas de entrada y salida de datos y modelos
data_dir = os.path.join(script_dir, '..', 'data', 'augmented')
data_file = os.path.join(data_dir, 'data_landmarks_sequences_augmented.pkl')

model_save_dir = os.path.join(script_dir, '..', 'models')
model_save_path = os.path.join(model_save_dir, 'word_model.h5')
label_encoder_save_path = os.path.join(model_save_dir, 'word_label_encoder.pkl')

os.makedirs(model_save_dir, exist_ok=True)

# Carga de datos
print(f"Cargando datos aumentados desde: {data_file}")
try:
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    sequences = data['sequences']
    labels = data['labels']
    print(f"Cargadas {len(sequences)} secuencias de landmarks con {len(set(labels))} clases.")
except FileNotFoundError:
    print(f"Error: El archivo de datos no se encontró en {data_file}.")
    print("Asegúrate de haber ejecutado 'extract_video_landmarks.py' y 'augment_landmarks.py' previamente.")
    exit()
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    exit()

# Preprocesamiento de datos
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
print(f"Clases detectadas: {label_encoder.classes_}")
print(f"Número total de clases: {num_classes}")

y_categorical = to_categorical(encoded_labels, num_classes=num_classes)

print("Normalizando la longitud de las secuencias...")
max_sequence_length = max(len(seq) for seq in sequences)
print(f"Longitud máxima de secuencia encontrada: {max_sequence_length}")

# Ajustar la forma de los datos para TimeDistributed(Conv1D)
# Las secuencias son (num_frames, num_features_per_frame)
# Necesitamos que Conv1D trate las 1629 características como una secuencia 1D.
# Por lo tanto, cada frame de (1629,) debe ser (1629, 1) para Conv1D.
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')

# Reshape para TimeDistributed Conv1D: (muestras, longitud_secuencia, características_por_frame, 1)
# O simplemente (muestras, longitud_secuencia, características_por_frame) si Conv1D acepta 2D input.
# Keras Conv1D espera entrada de 3D (batch, steps, features).
# TimeDistributed aplica una capa a cada paso temporal del input.
# Por lo tanto, si nuestro input es (muestras, max_len, 1629),
# TimeDistributed(Conv1D) tratará cada (1629,) como una "secuencia" para la Conv1D.
X = np.array(padded_sequences)
# Si tu Conv1D espera (features, 1) para un solo frame, tendríamos que reshapes X a (samples, max_len, 1629, 1)
# Pero por defecto, TimeDistributed(Conv1D) con input (samples, timesteps, features)
# aplica Conv1D a (features) para cada timestep. Esto es lo que queremos.


print(f"Forma de X (datos de entrada antes de CNN): {X.shape} (muestras, longitud_secuencia, características_por_frame)")
print(f"Forma de y (etiquetas one-hot): {y_categorical.shape} (muestras, num_clases)")


X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# Construcción del modelo CNN-LSTM
print("\nConstruyendo el modelo CNN-LSTM...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento del modelo
print("\nEntrenando el modelo CNN-LSTM...")
history = model.fit(X_train, y_train,
                    epochs=100, 
                    batch_size=32,
                    validation_split=0.1, 
                    verbose=1)

# Evaluación del modelo
print("\nEvaluando el modelo en el conjunto de prueba...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")

# Guardar el modelo y el LabelEncoder
print(f"\nGuardando el modelo de palabras en {model_save_path}")
model.save(model_save_path)

print(f"Guardando el Label Encoder de palabras en {label_encoder_save_path}")
joblib.dump(label_encoder, label_encoder_save_path)

print("Proceso de entrenamiento completado. ¡Tu modelo de reconocimiento de palabras está listo!")