import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input  # Import Input layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import seaborn as sns
import pywt
from scipy.fft import fft
from scipy.signal import hilbert
import time
from tensorflow.keras.callbacks import EarlyStopping


def preprocess_signal_SNR(signal, obj_snr_db, fs):
    """
    Preprocesa la señal añadiendo ruido gaussiano con un nivel SNR específico y extrayendo características.

    Args:
        signal (np.ndarray): La señal de entrada.
        obj_snr_db (int): El nivel de relación señal-ruido (SNR) en dB.
        fs (int): Frecuencia de muestreo

    Returns:
        np.ndarray: Un vector con las características extraídas de la señal con ruido.
    """
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-9)  # Añadido epsilon
    noise_avg_db = sig_avg_db - obj_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    noisy_signal = signal + noise_volts

    return preprocess_signal(noisy_signal, fs)


def preprocess_signal(signal, fs):
    """
    Calcula las características de la señal.

    Args:
        signal (np.ndarray): La señal de entrada.
        fs (int): Frecuencia de muestreo

    Returns:
        np.ndarray: Un vector con las características extraídas.
    """
    mean_value = np.mean(signal)
    unbias_data = signal - mean_value
    variance = np.var(unbias_data)
    skewness = np.mean(unbias_data ** 3) / (variance ** 1.5) if variance > 1e-9 else 0.0
    kurtosis = np.mean(unbias_data ** 4) / (variance ** 2) - 3 if variance > 1e-9 else 0.0

    fft_signal = np.fft.fft(signal)
    if len(fft_signal) > 3 and np.abs(fft_signal[1]) > 1e-9:
        thd_time_domain = np.sqrt(np.sum(np.abs(fft_signal[2:4]) ** 2)) / np.abs(fft_signal[1])
    else:
        thd_time_domain = 0.0

    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = np.max(signal) / (rms + 1e-9) if rms > 1e-9 else 0.0

    coeffs = pywt.wavedec(signal, 'db4', level=4)
    wavelet_features = []
    try:
        for c in coeffs:
            if len(c) > 0:
                wavelet_features.extend([np.mean(c), np.std(c)])
            else:
                wavelet_features.extend([0, 0])
    except ValueError as e:
        print(f"Error de Wavelet: {e}")
        wavelet_features = [0] * 10

    if len(wavelet_features) < 10:
        wavelet_features.extend([0] * (10 - len(wavelet_features)))
    elif len(wavelet_features) > 10:
        wavelet_features = wavelet_features[:10]
    wavelet_features = np.array(wavelet_features)

    N = len(signal)
    yf = fft(signal)
    xf = np.fft.fftfreq(N, 1 / fs)

    magnitudes = np.abs(yf[1:N // 2])
    frecuencias = xf[1:N // 2]
    total_energy = np.sum(magnitudes ** 2)

    band_limits = [50, 150, 250, 350, 450, fs // 2]
    band_energies = []
    for i in range(len(band_limits) - 1):
        start_freq = band_limits[i]
        end_freq = band_limits[i + 1]
        indices = np.where((frecuencias >= start_freq) & (frecuencias < end_freq))[0]
        band_energy = np.sum(magnitudes[indices] ** 2)
        band_energies.append(band_energy / (total_energy + 1e-9))

    fundamental_index = np.argmin(np.abs(frecuencias - 50)) if len(frecuencias) > 0 else -1
    harmonic_indices = np.where(frecuencias % 50 == 0)[0] if len(frecuencias) > 0 else []

    thd_freq_domain = 0.0
    if fundamental_index != -1 and len(harmonic_indices) > 1 and magnitudes[fundamental_index] > 1e-9:
        thd_freq_domain = np.sqrt(np.sum(magnitudes[harmonic_indices[1:]] ** 2)) / magnitudes[fundamental_index]

    num_magnitude_features = 11
    if len(magnitudes) < num_magnitude_features:
        selected_magnitudes = np.pad(magnitudes, (0, num_magnitude_features - len(magnitudes)), 'constant')
    else:
        selected_magnitudes = magnitudes[:num_magnitude_features]

    frequency_features = np.concatenate([selected_magnitudes, np.array(band_energies), np.array([thd_freq_domain])])

    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    envelope_max = np.max(envelope)
    envelope_features = np.array([envelope_mean, envelope_std, envelope_max])

    derivative = np.diff(signal)
    derivative = np.concatenate(([0], derivative))
    max_derivative = np.max(np.abs(derivative))
    mean_abs_derivative = np.mean(np.abs(derivative))
    derivative_features = np.array([max_derivative, mean_abs_derivative])

    return np.concatenate([
        np.array([variance, skewness, kurtosis, thd_time_domain, crest_factor]),
        wavelet_features,
        frequency_features,
        envelope_features,
        derivative_features
    ])


def load_signal(data_path, fs):
    label_mapping = {
        "flicker_signals": 0, "harmonic_signals": 1, "interruption_signals": 2,
        "original_signals": 3, "sag_signals": 4, "swell_signals": 5,
        "transient_signals": 6, "harmonic_sag_signals": 7,
        "harmonic_swell_signals": 8, "Harmonic_interruption_signals": 9,
    }
    features = []
    labels = []
    for signal_type, label in label_mapping.items():
        signal_type_path = os.path.join(data_path, signal_type)
        if os.path.isdir(signal_type_path):
            for root, _, files in os.walk(signal_type_path):
                for filename in files:
                    if filename.endswith(".npy"):
                        file_path = os.path.join(root, filename)
                        signal = np.load(file_path)
                        features.append(signal)
                        labels.append(label)
    return np.array(features), np.array(labels)


# --- Carga de datos ---
data_path = "data"
FS = 10000
original_features, original_labels = load_signal(data_path, FS)

print(f"Forma de las características originales cargadas: {original_features.shape}")
print(f"Forma de las etiquetas originales cargadas: {original_labels.shape}")

unique_labels, counts = np.unique(original_labels, return_counts=True)
print("Etiquetas únicas:", unique_labels)
print("Distribución de señales por categoría:")
for label, count in zip(unique_labels, counts):
    print(f"Clase {label}: {count} señales")

# --- División de datos (70/15/15) ---
X_train_clean, X_rem_clean, y_train_clean, y_rem_clean = train_test_split(original_features, original_labels,
                                                                          train_size=0.7, random_state=42,
                                                                          stratify=original_labels)
X_val_clean, X_test_clean, y_val_clean, y_test_clean = train_test_split(X_rem_clean, y_rem_clean, test_size=0.5,
                                                                        random_state=42, stratify=y_rem_clean)

print(f"Tamaño del conjunto de entrenamiento (limpio): {X_train_clean.shape[0]}")
print(f"Tamaño del conjunto de validación (limpio): {X_val_clean.shape[0]}")
print(f"Tamaño del conjunto de prueba (limpio): {X_test_clean.shape[0]}")

# --- Generación de ruido aleatorio para TRAIN y VALIDACIÓN ---
# Rango de SNR para el ruido aleatorio durante el entrenamiento y validación
train_val_snr_range = range(0, 41)  # Por ejemplo, de 0 dB a 40 dB

print("\nGenerando y preprocesando datos de entrenamiento y validación con ruido aleatorio...")

X_train_noisy_processed_list = []
for i, signal in enumerate(X_train_clean):
    random_snr_db = np.random.choice(train_val_snr_range)
    X_train_noisy_processed_list.append(preprocess_signal_SNR(signal, random_snr_db, FS))
X_train_noisy_processed = np.array(X_train_noisy_processed_list)

X_val_noisy_processed_list = []
for i, signal in enumerate(X_val_clean):
    random_snr_db = np.random.choice(train_val_snr_range)
    X_val_noisy_processed_list.append(preprocess_signal_SNR(signal, random_snr_db, FS))
X_val_noisy_processed = np.array(X_val_noisy_processed_list)

print("Generación y preprocesamiento de datos de entrenamiento y validación con ruido aleatorio completado.")

# --- Normalización de datos (ajustar escaladores solo en el TRAIN SET RUIDOSO) ---
scaler_time_domain = StandardScaler()
scaler_wavelet = StandardScaler()
scaler_frequency = StandardScaler()
scaler_envelope = StandardScaler()
scaler_derivative = RobustScaler()

# Ajustar los escaladores en el conjunto de entrenamiento RUIDOSO y transformar todos los conjuntos
X_train_time_domain = scaler_time_domain.fit_transform(X_train_noisy_processed[:, :5])
X_val_time_domain = scaler_time_domain.transform(X_val_noisy_processed[:, :5])
# OJO: X_test_clean se usará para generar las señales de prueba con ruido específico en el bucle
# No transformamos X_test_clean aquí, se hará en el bucle

X_train_wavelet = scaler_wavelet.fit_transform(X_train_noisy_processed[:, 5:15])
X_val_wavelet = scaler_wavelet.transform(X_val_noisy_processed[:, 5:15])

X_train_frequency = scaler_frequency.fit_transform(X_train_noisy_processed[:, 15:32])
X_val_frequency = scaler_frequency.transform(X_val_noisy_processed[:, 15:32])

X_train_envelope = scaler_envelope.fit_transform(X_train_noisy_processed[:, 32:35])
X_val_envelope = scaler_envelope.transform(X_val_noisy_processed[:, 32:35])

X_train_derivative = scaler_derivative.fit_transform(X_train_noisy_processed[:, 35:])
X_val_derivative = scaler_derivative.transform(X_val_noisy_processed[:, 35:])

# Concatenar las características normalizadas para TRAIN y VALIDACIÓN
X_train_normalized = np.concatenate([
    X_train_time_domain, X_train_wavelet, X_train_frequency,
    X_train_envelope, X_train_derivative
], axis=1)
X_val_normalized = np.concatenate([
    X_val_time_domain, X_val_wavelet, X_val_frequency,
    X_val_envelope, X_val_derivative
], axis=1)

# Convertir etiquetas a formato one-hot (usamos las etiquetas originales de los conjuntos, no las ruidosas)
y_train_one_hot = to_categorical(y_train_clean, num_classes=10)
y_val_one_hot = to_categorical(y_val_clean, num_classes=10)
y_test_one_hot = to_categorical(y_test_clean, num_classes=10)

# -------------------------------------------------------------
# DEFINICIÓN Y ENTRENAMIENTO DEL MODELO DNN (UNA SOLA VEZ CON DATOS RUIDOSOS)
# -------------------------------------------------------------

# Definir el modelo con la sugerencia de Keras para Input layer
model = Sequential([
    LSTM(37, activation='tanh', kernel_regularizer=l2(0.001), return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='tanh',return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='tanh',return_sequences=True),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    Dense(10, activation='sigmoid')  # Capa de salida para 10 clases
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Añadir Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nIniciando el entrenamiento del modelo DNN en datos con ruido aleatorio...")
# Entrenar el modelo
history = model.fit(X_train_normalized, y_train_one_hot,
                    validation_data=(X_val_normalized, y_val_one_hot),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1)
print("Entrenamiento del modelo DNN completado.")

# -------------------------------------------------------------
# EVALUACIÓN DEL MODELO DNN ENTRENADO EN DATOS CON DIFERENTES NIVELES DE RUIDO (DE dB EN dB)
# -------------------------------------------------------------

# Rangos de niveles de ruido a iterar para la evaluación
noise_levels_db = range(40, -1, -1)  # De 40 dB a 0 dB (inclusive)

# Listas para almacenar los resultados de cada iteración de evaluación
accuracies_snr = []
precisions_snr = []
recalls_snr = []
f1_scores_snr = []
noise_values_snr = []
times_snr_eval = []

print("\nIniciando la evaluación del modelo DNN en datos de prueba con ruido específico (dB en dB):")
for noise_db in noise_levels_db:
    print(f"\n--- Evaluando con nivel de ruido: {noise_db} dB ---")
    start_time_eval = time.time()

    # Generar señales de prueba con el ruido actual (X_test_clean es el conjunto de prueba original)
    X_test_noisy_signals_current_snr = np.array(
        [preprocess_signal_SNR(signal, noise_db, FS) for signal in X_test_clean])

    # Normalizar las características ruidosas del conjunto de prueba, usando los scalers ajustados en el TRAIN SET RUIDOSO
    X_test_SNR_time_domain = scaler_time_domain.transform(X_test_noisy_signals_current_snr[:, :5])
    X_test_SNR_wavelet = scaler_wavelet.transform(X_test_noisy_signals_current_snr[:, 5:15])
    X_test_SNR_frequency = scaler_frequency.transform(X_test_noisy_signals_current_snr[:, 15:32])
    X_test_SNR_envelope = scaler_envelope.transform(X_test_noisy_signals_current_snr[:, 32:35])
    X_test_SNR_derivative = scaler_derivative.transform(X_test_noisy_signals_current_snr[:, 35:])

    X_test_SNR_normalized_current_snr = np.concatenate([
        X_test_SNR_time_domain, X_test_SNR_wavelet, X_test_SNR_frequency,
        X_test_SNR_envelope, X_test_SNR_derivative
    ], axis=1)

    # Predecir sobre el conjunto de prueba ruidoso
    y_pred_noisy_current_snr = model.predict(X_test_SNR_normalized_current_snr, verbose=0)
    y_pred_classes_noisy_current_snr = np.argmax(y_pred_noisy_current_snr, axis=1)

    # Las etiquetas verdaderas para la evaluación del conjunto ruidoso son las originales de X_test_clean
    y_test_classes_for_noisy_eval = np.argmax(y_test_one_hot, axis=1)

    # Calcular métricas
    accuracy = accuracy_score(y_test_classes_for_noisy_eval, y_pred_classes_noisy_current_snr)
    report = classification_report(y_test_classes_for_noisy_eval, y_pred_classes_noisy_current_snr, output_dict=True,
                                   zero_division=0)

    precision_weighted = report['weighted avg']['precision']
    recall_weighted = report['weighted avg']['recall']
    f1_weighted = report['weighted avg']['f1-score']

    accuracies_snr.append(accuracy)
    precisions_snr.append(precision_weighted)
    recalls_snr.append(recall_weighted)
    f1_scores_snr.append(f1_weighted)
    noise_values_snr.append(noise_db)

    end_time_eval = time.time()
    iteration_time_eval = end_time_eval - start_time_eval
    times_snr_eval.append(iteration_time_eval)

    print(f"Accuracy para {noise_db} dB: {accuracy:.4f}")
    print(f"Precision (ponderada) para {noise_db} dB: {precision_weighted:.4f}")
    print(f"Recall (ponderado) para {noise_db} dB: {recall_weighted:.4f}")
    print(f"F1-score (ponderado) para {noise_db} dB: {f1_weighted:.4f}")
    print(f"Tiempo de computación para {noise_db} dB: {iteration_time_eval:.2f} segundos")

# -------------------------------------------------------------
# GRÁFICOS DE RESULTADOS DE LA EVALUACIÓN DE SNR
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(noise_values_snr, accuracies_snr, color='r', label='Accuracy')
plt.plot(noise_values_snr, precisions_snr, color='b', label='Precision (Weighted)')
plt.plot(noise_values_snr, recalls_snr, color='orange', label='Recall (Weighted)')
plt.plot(noise_values_snr, f1_scores_snr, color='black', label='F1-Score (Weighted)')
plt.xlabel('Nivel de Ruido SNR (dB)')
plt.ylabel('Métricas de Clasificación')
plt.title('Rendimiento del Modelo DNN vs. Nivel de Ruido (Entrenado con ruido aleatorio)')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

total_evaluation_time_snr = sum(times_snr_eval)
print(f"\nTiempo total de evaluación para todos los niveles de SNR: {total_evaluation_time_snr:.2f} segundos")

# -------------------------------------------------------------
# GRÁFICOS DE ENTRENAMIENTO DE LA DNN (PÉRDIDA Y PRECISIÓN POR ÉPOCAS)
# -------------------------------------------------------------
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida de Entrenamiento y Validación de la DNN (con ruido aleatorio)')
plt.show()

plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión de Entrenamiento y Validación de la DNN (con ruido aleatorio)')
plt.show()

# -------------------------------------------------------------
# MATRIZ DE CONFUSIÓN PARA UN CASO REPRESENTATIVO (EJ: 20dB SNR)
# -------------------------------------------------------------
# Para este gráfico, vamos a generar la matriz de confusión para un SNR específico (por ejemplo, 20 dB)
# Puedes cambiar este valor a cualquier otro dentro de tu rango de ruido.
representative_snr = 20
print(f"\nGenerando Matriz de Confusión para SNR de {representative_snr} dB...")

# Generar y normalizar el conjunto de prueba para el SNR representativo
X_test_noisy_signals_rep = np.array([preprocess_signal_SNR(signal, representative_snr, FS) for signal in X_test_clean])
X_test_SNR_normalized_rep_time_domain = scaler_time_domain.transform(X_test_noisy_signals_rep[:, :5])
X_test_SNR_normalized_rep_wavelet = scaler_wavelet.transform(X_test_noisy_signals_rep[:, 5:15])
X_test_SNR_normalized_rep_frequency = scaler_frequency.transform(X_test_noisy_signals_rep[:, 15:32])
X_test_SNR_normalized_rep_envelope = scaler_envelope.transform(X_test_noisy_signals_rep[:, 32:35])
X_test_SNR_normalized_rep_derivative = scaler_derivative.transform(X_test_noisy_signals_rep[:, 35:])

X_test_SNR_normalized_rep = np.concatenate([
    X_test_SNR_normalized_rep_time_domain, X_test_SNR_normalized_rep_wavelet, X_test_SNR_normalized_rep_frequency,
    X_test_SNR_normalized_rep_envelope, X_test_SNR_normalized_rep_derivative
], axis=1)

y_pred_rep = model.predict(X_test_SNR_normalized_rep, verbose=0)
y_pred_classes_rep = np.argmax(y_pred_rep, axis=1)
y_test_classes_rep = np.argmax(y_test_one_hot, axis=1)

matrix_rep = confusion_matrix(y_test_classes_rep, y_pred_classes_rep)

plt.figure(figsize=(16, 7))
sns.set(font_scale=1)
sns.heatmap(matrix_rep, annot=True, annot_kws={'size': 10}, fmt='d', cmap=plt.cm.Blues, linewidths=0.5)

class_names = ['Flicker', 'Harmónico', 'Interrupción', 'Señal Original', 'Sag', 'Swell', 'Transitorio',
               "Harmónico + Sag", "Harmónico + Swell", "Harmónico + Interruption"]
tick_marks = np.arange(len(class_names)) + 0.5
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=30)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Etiqueta Predicha', weight='bold', fontsize=14)
plt.ylabel('Clase Real', weight='bold', fontsize=14)
plt.title(f'Matriz de Confusión para el Modelo DNN (SNR: {representative_snr} dB)')
plt.show()