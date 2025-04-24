from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def preprocess_signal(signal, obj_snr_db):
    """
    Preprocesa la señal añadiendo ruido con un nivel SNR específico y extrayendo características.

    Args:
        signal (np.ndarray): La señal de entrada.
        obj_snr_db (int): El nivel de relación señal-ruido (SNR) en dB.

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

    mean_value = np.mean(noisy_signal)
    unbias_data = noisy_signal - mean_value
    variance = np.var(unbias_data)
    skewness = np.mean(unbias_data ** 3) / (variance ** 1.5)
    kurtosis = np.mean(unbias_data ** 4) / (variance ** 2) - 3
    thd = np.sqrt(np.sum(np.abs(np.fft.fft(noisy_signal)[2:4])) / np.abs(np.fft.fft(noisy_signal)[1]))
    rms = np.sqrt(np.mean(noisy_signal ** 2))
    crest_factor = np.max(noisy_signal) / rms
    return np.array([variance, skewness, kurtosis, thd, crest_factor])

def load_signal(data_path):
    """
    Carga las señales desde los directorios y asigna etiquetas.

    Args:
        data_path (str): La ruta al directorio principal de los datos.

    Returns:
        tuple: Un array con las características y un array con las etiquetas.
    """
    label_mapping = {
        "flicker_signals": 0,
        "harmonic_signals": 1,
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
                            # La preprocesamiento ahora se hará dentro del bucle de iteración
                            features.append(signal) # Guardamos la señal original para añadir ruido después
                            labels.append(label)
    return np.array(features), np.array(labels)

# Ejemplo de uso de la carga de datos
data_path = "data"  # ¡Asegúrate de que esta ruta sea correcta para tu sistema!
original_features, original_labels = load_signal(data_path)

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

for noise_db in noise_levels_db:
    print(f"\n--- Procesando con nivel de ruido: {noise_db} dB ---")

    # Preprocesar las señales originales con el nivel de ruido actual
    noisy_features = np.array([preprocess_signal(signal, noise_db) for signal in original_features])

    # Dividir los datos en conjunto de entrenamiento y prueba (usando las características con el ruido actual)
    X_train, X_test, y_train, y_test = train_test_split(noisy_features, original_labels, test_size=0.2, random_state=1, stratify=original_labels)

    # Definir y entrenar el modelo (puedes mantener tu búsqueda de hiperparámetros si lo deseas)
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['gini', 'entropy']
    }
    dt_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy') # Cambié scoring a 'accuracy' para la iteración
    grid_search.fit(X_train, y_train)
    best_dt_model = grid_search.best_estimator_
    y_pred = best_dt_model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Calcular la media de precisión, recall y F1-score (ponderados por la cantidad de muestras en cada clase)
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

# Generar la gráfica
plt.figure(figsize=(10, 6))
plt.plot(noise_values, accuracies, color = 'r', label='Accuracy')
plt.plot(noise_values, precisions, color = 'b', label='Precision (Weighted)')
plt.plot(noise_values, recalls, color = 'orange',label='Recall (Weighted)')
plt.plot(noise_values, f1_scores, color = 'black', label='F1-Score (Weighted)')
plt.xlabel('Nivel de Ruido (dB)')
plt.ylabel('Métricas de Clasificación')
plt.title('Rendimiento del Modelo vs. Nivel de Ruido')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis() # Invertir el eje x para que el ruido disminuya de izquierda a derecha
plt.show()