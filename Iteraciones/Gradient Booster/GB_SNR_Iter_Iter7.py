import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pywt
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

def preprocess_signal_SNR(signal, obj_snr_db, fs):
    """
    Preprocesa la señal añadiendo ruido con un nivel SNR específico y extrayendo características.

    Args:
        signal (np.ndarray): La señal de entrada.
        obj_snr_db (int): El nivel de relación señal-ruido (SNR) en dB.
        fs (int): Frecuencia de muestreo

    Returns:
        np.ndarray: Un vector con las características extraídas de la señal con ruido.
    """
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
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
    skewness = np.mean(unbias_data ** 3) / (variance ** 1.5)
    kurtosis = np.mean(unbias_data ** 4) / (variance ** 2) - 3
    thd_time_domain = np.sqrt(np.sum(np.abs(np.fft.fft(signal)[2:4])) / np.abs(np.fft.fft(signal)[1]))
    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = np.max(signal) / rms

    # Wavelet features
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

    wavelet_features = np.array(wavelet_features)

    # Frequency domain features
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
        band_energies.append(band_energy / total_energy)

    fundamental_index = np.argmin(np.abs(frecuencias - 50))
    harmonic_indices = np.where(frecuencias % 50 == 0)[0]
    if len(harmonic_indices) > 1:
        thd_freq_domain = np.sqrt(np.sum(magnitudes[harmonic_indices[1:]] ** 2)) / magnitudes[fundamental_index]
    else:
        thd_freq_domain = 0

    frequency_features = np.concatenate([magnitudes, np.array(band_energies), np.array([thd_freq_domain])])

    all_features = np.concatenate([np.array([variance, skewness, kurtosis, thd_time_domain, crest_factor]), wavelet_features, frequency_features])

    return all_features

def load_signal(data_path, fs):
    """
    Carga las señales desde los directorios y asigna etiquetas.

    Args:
        data_path (str): La ruta al directorio principal de los datos.
        fs (int): Frecuencia de muestreo

    Returns:
        tuple: Un array con las características y un array con las etiquetas.
    """
    label_mapping = {
        #"flicker_signals": 0,
        #"harmonic_signals": 1,
        "interruption_signals": 2,
        "original_signals": 3,
        "sag_signals": 4,
        "swell_signals": 5,
        "transient_signals": 6,
        "harmonic_sag_signals": 7,
        "harmonic_swell_signals": 8,
        "Harmonic_interruption_signals": 9,
    }
    features = []
    labels = []
    for signal_type, label in label_mapping.items():
        signal_type_path = os.path.join(data_path, signal_type)
        if os.path.isdir(signal_type_path):
            for subset in ["train", "test", "val"]:
                subset_path = os.path.join(signal_type_path, subset)
                if os.path.exists(subset_path):
                    for filename in os.listdir(subset_path):
                        if filename.endswith(".npy"):
                            file_path = os.path.join(subset_path, filename)
                            signal = np.load(file_path)
                            features.append(signal)
                            labels.append(label)
    return np.array(features), np.array(labels)

# Ejemplo de uso de la carga de datos
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

if 3 not in unique_labels:
    print("Error: La clase 3 no está presente en los datos originales.")

# Rangos de niveles de ruido a iterar
noise_levels_db = range(40, 0, -1)

# Listas para almacenar los resultados de cada iteración
accuracies = []
precisions = []
recalls = []
f1_scores = []
noise_values = []

# Inicializar el StandardScaler
scaler = StandardScaler()

for noise_db in noise_levels_db:
    print(f"\n--- Procesando con nivel de ruido: {noise_db} dB ---")

    # Preprocesar las señales
    noisy_features = np.array([preprocess_signal_SNR(signal, noise_db, FS) for signal in original_features])
    features_1 = np.array([preprocess_signal(signal, FS) for signal in original_features])

    # Normalizar los datos
    X_train, X_test, y_train, y_test = train_test_split(features_1, original_labels, test_size=0.2, random_state=42,
                                                        stratify=original_labels)
    X_train_SNR, X_test_SNR, y_train_SNR, y_test_SNR = train_test_split(noisy_features, original_labels, test_size=0.2, random_state=42,
                                                        stratify=original_labels)

    # Ajustar el scaler solo a los datos de entrenamiento para evitar fuga de información
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_SNR = scaler.transform(X_train_SNR)
    X_test_SNR = scaler.transform(X_test_SNR)

    # Crear el modelo de Gradient Boosting
    GradBoost = HistGradientBoostingClassifier(
        max_iter=100,
        loss='log_loss',
        random_state=123,
        l2_regularization=0.1  # Añadimos regularización L2
    )

    # Entrenar el modelo
    GradBoost.fit(X_train, y_train)

    # Realizar predicciones
    predicciones = GradBoost.predict(X_test_SNR)

    # Calcular métricas
    accuracy = accuracy_score(y_test_SNR, predicciones)
    y_pred = GradBoost.predict(X_test_SNR)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    precision_weighted = report['weighted avg']['precision']
    recall_weighted = report['weighted avg']['recall']
    f1_weighted = report['weighted avg']['f1-score']

    accuracies.append(accuracy)
    precisions.append(precision_weighted)
    recalls.append(recall_weighted)
    f1_scores.append(f1_weighted)
    noise_values.append(noise_db)

    print(f"Accuracy para {noise_db} dB: {accuracy:.2f}")
    print(f"Precision (ponderada) para {noise_db} dB: {precision_weighted:.2f}")
    print(f"Recall (ponderado) para {noise_db} dB: {recall_weighted:.2f}")
    print(f"F1-score (ponderado) para {noise_db} dB: {f1_weighted:.2f}")

    # Calcular y mostrar la matriz de confusión
    cm = confusion_matrix(y_test_SNR, predicciones)
    print("Matriz de Confusión:")
    print(cm)

# Generar la gráfica de métricas vs. SNR
plt.figure(figsize=(10, 6))
plt.plot(noise_values, accuracies, color='r', label='Accuracy')
plt.plot(noise_values, precisions, color='b', label='Precision (Weighted)')
plt.plot(noise_values, recalls, color='orange', label='Recall (Weighted)')
plt.plot(noise_values, f1_scores, color='black', label='F1-Score (Weighted)')
plt.xlabel('Nivel de Ruido SNR (dB)')
plt.ylabel('Métricas de Clasificación')
plt.title('Rendimiento del Modelo vs. Nivel de Ruido para Gradient Booster')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()