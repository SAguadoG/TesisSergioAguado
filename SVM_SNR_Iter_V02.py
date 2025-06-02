import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pywt
from scipy.fft import fft
from scipy.signal import hilbert
import time

def preprocess_signal_SNR(signal, obj_snr_db, fs):

# En esta parte lo que se hace es añadir un ruido aleatorio para el procesamiento de señales para el entrenamiento.

    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-9)  # Epsilon para evitar log(0)
    noise_avg_db = sig_avg_db - obj_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    noisy_signal = signal + noise_volts
    return preprocess_signal(noisy_signal, fs)

def preprocess_signal(signal, fs):

# En esta parte se extraen las características de las señales.

# CARACTERISTICAS EN EL DOMINIO DEL TIEMPO (HOS)

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
    time_domain_features = np.array([variance, skewness, kurtosis, thd_time_domain, crest_factor])

# CARACTERÍSTICAS WAVELET

    coeffs = pywt.wavedec(signal, 'db4', level=4)
    wavelet_features = []
    for c in coeffs:
        wavelet_features.extend([np.mean(c), np.std(c) if len(c) > 1 else 0])
    wavelet_features = np.nan_to_num(np.array(wavelet_features, dtype=float))
    if len(wavelet_features) > 10:
        wavelet_features = wavelet_features[:10]
    else:
        wavelet_features = np.pad(wavelet_features, (0, 10 - len(wavelet_features)), 'constant')

# CARACTERÍSTICAS EN EL DOMINIO DE LA FRECUENCIA.

    N = len(signal)
    yf = fft(signal)
    xf = np.fft.fftfreq(N, 1 / fs)
    magnitudes = np.abs(yf[1:N // 2])
    frecuencias = xf[1:N // 2]
    total_energy = np.sum(magnitudes ** 2)

    band_limits = [50, 150, 250, 350, 450, fs // 2]
    band_energies = []
    for i in range(len(band_limits) - 1):
        start_freq, end_freq = band_limits[i], band_limits[i + 1]
        indices = np.where((frecuencias >= start_freq) & (frecuencias < end_freq))[0]
        band_energy = np.sum(magnitudes[indices] ** 2)
        band_energies.append(band_energy / (total_energy + 1e-9))

    num_magnitude_features = 11
    if len(magnitudes) < num_magnitude_features:
        selected_magnitudes = np.pad(magnitudes, (0, num_magnitude_features - len(magnitudes)), 'constant')
    else:
        selected_magnitudes = magnitudes[:num_magnitude_features]

    frequency_features = np.concatenate(
        [selected_magnitudes, np.array(band_energies), np.zeros(1)])  # THD en frecuencia se omite para simplicidad

# CARACTERÍSTICAS DE LA ENVOLVENTE

    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope_features = np.array([np.mean(envelope), np.std(envelope), np.max(envelope)])

# CARACTERÍSTICAS DE LA DERIVADA

    derivative = np.diff(signal, prepend=signal[0])
    derivative_features = np.array([np.max(np.abs(derivative)), np.mean(np.abs(derivative))])

# UNIMOS TODAS LAS CARACTERÍSTICAS.

    final_features = np.concatenate([
        time_domain_features,  # 5
        wavelet_features,  # 10
        frequency_features,  # 17 (11 mags + 5 bandas + 1 placeholder)
        envelope_features,  # 3
        derivative_features  # 2
    ])
    return np.nan_to_num(final_features)

def load_signal(data_path):

    label_mapping = {
        "flicker_signals": 0, "harmonic_signals": 1, "interruption_signals": 2,
        "original_signals": 3, "sag_signals": 4, "swell_signals": 5,
        "transient_signals": 6, "harmonic_sag_signals": 7,
        "harmonic_swell_signals": 8, "Harmonic_interruption_signals": 9,
    }
    signals, labels = [], []
    for signal_type, label in label_mapping.items():
        signal_type_path = os.path.join(data_path, signal_type)
        if os.path.isdir(signal_type_path):
            for root, _, files in os.walk(signal_type_path):
                for filename in files:
                    if filename.endswith(".npy"):
                        file_path = os.path.join(root, filename)
                        signal = np.load(file_path)
                        signals.append(signal)
                        labels.append(label)
    return np.array(signals, dtype=object), np.array(labels)

# CARGA Y DIVISIÓN DE DATOS (80/20)

data_path = "data"
FS = 10000
original_signals, original_labels = load_signal(data_path)

# División 80% train, 20% test
X_train_clean, X_test_clean, y_train, y_test = train_test_split(
    original_signals, original_labels, train_size=0.7, random_state=42, stratify=original_labels)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train_clean)}")
print(f"Tamaño del conjunto de prueba: {len(X_test_clean)}")

train_val_snr_range = range(0, 41)
train_time_start = time.time()
print("\nGenerando datos de entrenamiento y validación con ruido aleatorio...")
X_train_noisy = np.array([preprocess_signal_SNR(s, np.random.choice(train_val_snr_range), FS) for s in X_train_clean])

# NORMALIZACIÓN

scaler_time = StandardScaler().fit(X_train_noisy[:, :5])
scaler_wavelet = StandardScaler().fit(X_train_noisy[:, 5:15])
scaler_freq = StandardScaler().fit(X_train_noisy[:, 15:32])
scaler_env = StandardScaler().fit(X_train_noisy[:, 32:35])
scaler_deriv = RobustScaler().fit(X_train_noisy[:, 35:])

X_train_normalized = np.concatenate([
    scaler_time.transform(X_train_noisy[:, :5]),
    scaler_wavelet.transform(X_train_noisy[:, 5:15]),
    scaler_freq.transform(X_train_noisy[:, 15:32]),
    scaler_env.transform(X_train_noisy[:, 32:35]),
    scaler_deriv.transform(X_train_noisy[:, 35:])
], axis=1)

# ENTRENAMIENTO DEL MODELO

print("\nIniciando el entrenamiento del modelo SVM en datos con ruido aleatorio...")
svm = SVC(C=100.0, kernel='rbf', gamma='scale', class_weight=None)
svm.fit(X_train_normalized, y_train)
print("Entrenamiento completado.")
train_time_end = time.time()
train_time = train_time_end - train_time_start

print(f"Tiempo de computación del Entrenamiento: {train_time:.2f} segundos")

# EVALUACIÓN DEL MODELO CON DIFERENTES NIVELES DE RUIDO

noise_levels_db = range(40, -1, -1)
results = []
times = []

print("\nIniciando la evaluación del modelo en datos de prueba con ruido específico...")
for noise_db in noise_levels_db:
    start_time_eval = time.time()

    # 1. Generar características del TEST SET con el ruido actual
    X_test_current_noise = np.array([preprocess_signal_SNR(s, noise_db, FS) for s in X_test_clean])

    # 2. Normalizar usando los scalers ya ajustados
    X_test_normalized = np.concatenate([
        scaler_time.transform(X_test_current_noise[:, :5]),
        scaler_wavelet.transform(X_test_current_noise[:, 5:15]),
        scaler_freq.transform(X_test_current_noise[:, 15:32]),
        scaler_env.transform(X_test_current_noise[:, 32:35]),
        scaler_deriv.transform(X_test_current_noise[:, 35:])
    ], axis=1)

    # 3. Predecir y calcular métricas
    y_pred = svm.predict(X_test_normalized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results.append({
        'snr': noise_db,
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    })
    print(f"--- SNR: {noise_db} dB | Accuracy: {accuracy:.4f} | F1-Score: {report['weighted avg']['f1-score']:.4f} ---")
    print(f"--- SNR: {noise_db} dB | Recall: {report['weighted avg']['recall']:.4f} | Precision: {report['weighted avg']['precision']:.4f} ---")
    end_time = time.time()
    iteration_time = end_time - start_time_eval
    times.append(iteration_time)
    print(f"Tiempo de computación para {noise_db} dB: {iteration_time:.2f} segundos")

# GRÁFICOS DE RESULTADOS

snr_values = [r['snr'] for r in results]
accuracies = [r['accuracy'] for r in results]
precisions = [r['precision'] for r in results]
f1_scores = [r['f1-score'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(snr_values, accuracies, color='b', label='Accuracy')
plt.plot(snr_values, precisions, color='r', label='Precision')
plt.plot(snr_values, f1_scores, color='black', label='F1-Score')
plt.xlabel('Nivel de Ruido SNR (dB) en el Conjunto de Prueba')
plt.ylabel('Métricas')
plt.title('Rendimiento de SVM vs. Nivel de Ruido')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

total_time = sum(times)
print(f"Tiempo total de computación: {total_time:.2f} segundos")