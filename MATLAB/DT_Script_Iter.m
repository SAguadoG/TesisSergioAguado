clear all;
clc;

% CARGA Y DIVISIÓN DE DATOS (80/20)

% data_path = "data_mat" es porque el script está en la misma carpeta que
% la carpeta principal de los datos, en el caso de no estarlo, habría que
% ajustarlo a la ruta.

data_path = "data_mat";
FS = 10000; 

% Se hace la llamada al script anexo de carga de señales, donde se extraen
% las señales y las etiquetas.
% las he llamado "original_signals" porque aun no han sido alteradas.

[original_signals, original_labels] = load_signal(data_path);

% El propio "cvpartition" da por hecho de que va a haber dos lineas a
% continuación, una de training y otra de test.

c = cvpartition(original_labels, 'Holdout', 0.2, 'Stratify', true);
train_idx = training(c);
test_idx = test(c);

X_train_clean = original_signals(train_idx);
X_test_clean = original_signals(test_idx);
y_train = original_labels(train_idx);
y_test = original_labels(test_idx);

fprintf('Tamaño del conjunto de entrenamiento: %d\n', length(X_train_clean));
fprintf('Tamaño del conjunto de prueba: %d\n', length(X_test_clean));

train_val_snr_range = 0:40;
train_time_start = tic;

fprintf('\nGenerando datos de entrenamiento y validación con ruido aleatorio\n');

num_features = 35;

X_train_noisy = zeros(length(X_train_clean), num_features);

% Barra de progreso para la generación de datos de entrenamiento
% Pongo la barra de progreso para evitar un spam masivo en la ventana de
% salida.

total_train_signals = length(X_train_clean);
progressbar('Generando características de entrenamiento');

for i = 1:total_train_signals

    rand_snr_idx = randi(length(train_val_snr_range));
    random_snr = train_val_snr_range(rand_snr_idx);
    X_train_noisy(i, :) = preprocess_signal_SNR(X_train_clean{i}, random_snr, FS);

    % Actualizar la barra de progreso
    progressbar(i / total_train_signals);

end

% NORMALIZACIÓN

scaler_time_mean = mean(X_train_noisy(:, 1:5));
scaler_time_std = std(X_train_noisy(:, 1:5));
X_train_noisy_scaled_time = (X_train_noisy(:, 1:5) - scaler_time_mean) ./ scaler_time_std;

scaler_wavelet_mean = mean(X_train_noisy(:, 6:13));
scaler_wavelet_std = std(X_train_noisy(:, 6:13));
X_train_noisy_scaled_wavelet = (X_train_noisy(:, 6:13) - scaler_wavelet_mean) ./ scaler_wavelet_std;

scaler_freq_mean = mean(X_train_noisy(:, 14:30));
scaler_freq_std = std(X_train_noisy(:, 14:30));
X_train_noisy_scaled_freq = (X_train_noisy(:, 14:30) - scaler_freq_mean) ./ scaler_freq_std;

scaler_env_mean = mean(X_train_noisy(:, 31:33));
scaler_env_std = std(X_train_noisy(:, 31:33));
X_train_noisy_scaled_env = (X_train_noisy(:, 31:33) - scaler_env_mean) ./ scaler_env_std;

scaler_deriv_median = median(X_train_noisy(:, 34:35));
scaler_deriv_iqr = iqr(X_train_noisy(:, 34:35));
scaler_deriv_iqr(scaler_deriv_iqr == 0) = 1e-9;
X_train_noisy_scaled_deriv = (X_train_noisy(:, 34:35) - scaler_deriv_median) ./ scaler_deriv_iqr;

% Unimos todo

X_train_normalized = [
    X_train_noisy_scaled_time, ...
    X_train_noisy_scaled_wavelet, ...
    X_train_noisy_scaled_freq, ...
    X_train_noisy_scaled_env, ...
    X_train_noisy_scaled_deriv
];


% ENTRENAMIENTO DEL MODELO

fprintf('\nIniciando el entrenamiento del modelo (Árbol de Decisión) en datos con ruido aleatorio\n');

DecisionTree = fitctree(X_train_normalized, y_train, ...
    'MinLeafSize', 1); 

fprintf('Entrenamiento del Árbol de Decisión completado.\n');
train_time = toc(train_time_start);
fprintf('Tiempo de computación del Entrenamiento: %.2f segundos\n', train_time);

fprintf('Entrenamiento completado.\n');
fprintf('Tiempo de computación del Entrenamiento: %.2f segundos\n', train_time);

% EVALUACIÓN DEL MODELO CON DIFERENTES NIVELES DE RUIDO

noise_levels_db = 40:-1:0;
num_noise_levels = length(noise_levels_db); % Obtenemos el número total de iteraciones

% Iniciamos todos los valores.
result_template = struct('snr', 0, ...
                         'accuracy', 0, ...
                         'precision', 0, ...
                         'recall', 0, ...
                         'f1_score', 0);

results = repmat(result_template, num_noise_levels, 1);
times = zeros(1, num_noise_levels); % Pre-asignamos 'times' también

fprintf(['\nIniciando la evaluación del modelo en datos de' ...
    'prueba de ruido SNR descente\n']);

% Metemos una barra de progreso, aunque también saldrán los resultados,
% como no requiere mucha computación y queda bien para saber por donde vas.
% se la he añadido a esta parte también.

progressbar('Evaluando el modelo con diferentes niveles de ruido');

current_noise_idx = 0; % Inicializamos un contador para indexar 'results' y 'times'

for noise_db = noise_levels_db
    current_noise_idx = current_noise_idx + 1;
    start_time_eval = tic;

    X_test_current_noise = zeros(length(X_test_clean), num_features);
    for i = 1:length(X_test_clean)
        X_test_current_noise(i, :) = preprocess_signal_SNR(X_test_clean{i}, noise_db, FS);
    end

    X_test_normalized_time = (X_test_current_noise(:, 1:5) - scaler_time_mean) ./ scaler_time_std;
    X_test_normalized_wavelet = (X_test_current_noise(:, 6:13) - scaler_wavelet_mean) ./ scaler_wavelet_std;
    X_test_normalized_freq = (X_test_current_noise(:, 14:30) - scaler_freq_mean) ./ scaler_freq_std;
    X_test_normalized_env = (X_test_current_noise(:, 31:33) - scaler_env_mean) ./ scaler_env_std;
    X_test_normalized_deriv = (X_test_current_noise(:, 34:35) - scaler_deriv_median) ./ scaler_deriv_iqr;
    
    X_test_normalized = [
        X_test_normalized_time, ...
        X_test_normalized_wavelet, ...
        X_test_normalized_freq, ...
        X_test_normalized_env, ...
        X_test_normalized_deriv
    ];
    
    % --------------------------------------------------------------------

    % INICIO DE DEBUGGING para la función predict
    
    % Esta sección se la pedí a Gemini para que me de hiciera una flag
    % ya que tenía problemas a la hora de que funcionara, es totalmente
    % prescindible.
    
    disp('--- Información de variables antes de la llamada a predict ---');
    disp(['Clase de DecisionTree: ' class(DecisionTree)]);
    disp(['¿DecisionTree está vacío?: ' mat2str(isempty(DecisionTree))]);
    if ~isempty(DecisionTree) && isobject(DecisionTree) && ismethod(DecisionTree, 'predict')
        disp('DecisionTree parece ser un objeto y tiene un método predict.');
    else
        disp('ADVERTENCIA: DecisionTree NO es un objeto válido o no tiene un método predict esperado.');
    end
    disp(['Clase de X_test_normalized: ' class(X_test_normalized)]);
    disp(['Dimensiones de X_test_normalized: ' mat2str(size(X_test_normalized))]);
    
    % --- Nuevas líneas de depuración para 'results' ---
    disp('--- Estado de la variable results antes de la asignación ---');
    disp(['Clase de results: ' class(results)]);
    disp(['Número de elementos en results: ' mat2str(length(results))]);
    if ~isempty(results)
        disp('Nombres de campos de results (primer elemento):');
        disp(fieldnames(results(1)));
    end
    disp('--------------------------------------------------------------');

    % --------------------------------------------------------------------

    % 3. Predecir y calcular métricas
    y_pred = predict(DecisionTree, X_test_normalized);

    y_test_numeric = double(y_test);
    y_pred_numeric = double(y_pred);

    accuracy = sum(y_pred_numeric == y_test_numeric) / length(y_test_numeric);

    [C, order] = confusionmat(y_test_numeric, y_pred_numeric);
    num_classes = size(C, 1);
    precision_per_class = zeros(1, num_classes);
    recall_per_class = zeros(1, num_classes);
    f1_per_class = zeros(1, num_classes);
    support_per_class = sum(C, 2);

    for c_idx = 1:num_classes
        TP = C(c_idx, c_idx);
        FP = sum(C(:, c_idx)) - TP;
        FN = sum(C(c_idx, :)) - TP;

        precision_per_class(c_idx) = TP / (TP + FP);
        recall_per_class(c_idx) = TP / (TP + FN);
        f1_per_class(c_idx) = 2 * (precision_per_class(c_idx) * recall_per_class(c_idx)) / (precision_per_class(c_idx) + recall_per_class(c_idx));

        if isnan(precision_per_class(c_idx)), precision_per_class(c_idx) = 0; end
        if isnan(recall_per_class(c_idx)), recall_per_class(c_idx) = 0; end
        if isnan(f1_per_class(c_idx)), f1_per_class(c_idx) = 0; end
    end

    weighted_precision = sum(precision_per_class .* (support_per_class' / sum(support_per_class)));
    weighted_recall = sum(recall_per_class .* (support_per_class' / sum(support_per_class)));
    weighted_f1_score = sum(f1_per_class .* (support_per_class' / sum(support_per_class)));

    results(current_noise_idx).snr = noise_db;
    results(current_noise_idx).accuracy = accuracy;
    results(current_noise_idx).precision = weighted_precision;
    results(current_noise_idx).recall = weighted_recall;
    results(current_noise_idx).f1_score = weighted_f1_score;

    fprintf('--- SNR: %d dB | Accuracy: %.4f | F1-Score: %.4f ---\n', noise_db, accuracy, weighted_f1_score);
    fprintf('--- SNR: %d dB | Recall: %.4f | Precision: %.4f ---\n', noise_db, weighted_recall, weighted_precision);
    iteration_time = toc(start_time_eval);
    times(current_noise_idx) = iteration_time;
    fprintf('Tiempo de computación para %d dB: %.2f segundos\n', noise_db, iteration_time);
    
    % Actualizar la barra de progreso dentro del bucle
    progressbar(current_noise_idx / num_noise_levels);

end

% GRÁFICOS DE RESULTADOS

snr_values = [results.snr];
accuracies = [results.accuracy];
precisions = [results.precision];
f1_scores = [results.f1_score];

figure;
plot(snr_values, accuracies, 'b', 'DisplayName', 'Accuracy');
hold on;
plot(snr_values, precisions, 'r', 'DisplayName', 'Precision');
plot(snr_values, f1_scores, 'k', 'DisplayName', 'F1-Score');
xlabel('Nivel de Ruido SNR (dB) en el Conjunto de Prueba');
ylabel('Métricas');
title('Rendimiento de Árbol de Decisión vs. Nivel de Ruido');
legend('Location', 'best');
grid on;

% Importarte invertir eje para que se vea una decadencia del rendimiento.
set(gca, 'XDir', 'reverse'); % Invertir eje

total_time = sum(times);
fprintf('Tiempo total de computación: %.2f segundos\n', total_time);