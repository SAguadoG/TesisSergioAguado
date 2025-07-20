clear all;
clc;


data_path = "data_mat";
FS = 10000;

[original_signals, original_labels] = load_signal(data_path);

c = cvpartition(original_labels, 'Holdout', 0.2, 'Stratify', true);
train_idx = training(c);
test_idx = test(c);

X_train_clean = original_signals(train_idx);
X_test_clean = original_signals(test_idx);
y_train = original_labels(train_idx); % Estas etiquetas son numéricas (0-9 si es el caso)
y_test = original_labels(test_idx);

fprintf('Tamaño del conjunto de entrenamiento: %d\n', length(X_train_clean));
fprintf('Tamaño del conjunto de prueba: %d\n', length(X_test_clean));

% GENERACIÓN DE DATOS DE ENTRENAMIENTO CON RUIDO ALEATORIO

fprintf('Generando datos de entrenamiento y validación con ruido aleatorio para el entrenamiento...\n');
tic;

num_features = 35;
X_train_noisy = zeros(length(X_train_clean), num_features);

total_train_signals = length(X_train_clean);
progressbar('Generando características de entrenamiento para DNN'); % Inicializar barra de progreso


for i = 1:length(X_train_clean)
    random_snr = randi([0, 40]);
    X_train_noisy(i, :) = preprocess_signal_SNR(X_train_clean{i}, random_snr, FS);

    % Actualizar la barra de progreso
    progressbar(i / total_train_signals);

end
fprintf('Generación de datos de entrenamiento y validación completada.\n');
generation_time = toc;
fprintf('Tiempo de computación de Generación: %.2f segundos\n', generation_time);

% Normalización

scaler_time_mean = mean(X_train_noisy(:, 1:5));
scaler_time_std = std(X_train_noisy(:, 1:5));
scaler_time_std(scaler_time_std == 0) = 1e-9;

scaler_wavelet_mean = mean(X_train_noisy(:, 6:13));
scaler_wavelet_std = std(X_train_noisy(:, 6:13));
scaler_wavelet_std(scaler_wavelet_std == 0) = 1e-9;

scaler_freq_mean = mean(X_train_noisy(:, 14:30));
scaler_freq_std = std(X_train_noisy(:, 14:30));
scaler_freq_std(scaler_freq_std == 0) = 1e-9;

scaler_env_mean = mean(X_train_noisy(:, 31:33));
scaler_env_std = std(X_train_noisy(:, 31:33));
scaler_env_std(scaler_env_std == 0) = 1e-9;

scaler_deriv_median = median(X_train_noisy(:, 34:35));
scaler_deriv_iqr = iqr(X_train_noisy(:, 34:35));
scaler_deriv_iqr(scaler_deriv_iqr == 0) = 1e-9;

X_train_normalized_time = (X_train_noisy(:, 1:5) - scaler_time_mean) ./ scaler_time_std;
X_train_normalized_wavelet = (X_train_noisy(:, 6:13) - scaler_wavelet_mean) ./ scaler_wavelet_std;
X_train_normalized_freq = (X_train_noisy(:, 14:30) - scaler_freq_mean) ./ scaler_freq_std;
X_train_normalized_env = (X_train_noisy(:, 31:33) - scaler_env_mean) ./ scaler_env_std;
X_train_normalized_deriv = (X_train_noisy(:, 34:35) - scaler_deriv_median) ./ scaler_deriv_iqr;

X_train_normalized = [
    X_train_normalized_time, ...
    X_train_normalized_wavelet, ...
    X_train_normalized_freq, ...
    X_train_normalized_env, ...
    X_train_normalized_deriv
];

rng(42); % Semilla para reproducibilidad
[train_dnn_ind, val_dnn_ind, ~] = dividerand(size(X_train_normalized, 1), 0.8, 0.2, 0); % 80% train, 20% val de X_train_normalized

X_dnn_train = X_train_normalized(train_dnn_ind, :);
y_dnn_train = y_train(train_dnn_ind); % Etiquetas numéricas

X_dnn_val = X_train_normalized(val_dnn_ind, :);
y_dnn_val = y_train(val_dnn_ind); % Etiquetas numéricas


% DNN

num_classes = length(unique(y_train));

% Convertir etiquetas a categorical para el entrenamiento de la DNN

y_dnn_train_cat = categorical(y_dnn_train);
y_dnn_val_cat = categorical(y_dnn_val);

layers = [
    featureInputLayer(num_features, 'Normalization', 'zscore')  % Capa de entrada con 35 características
    
    % Capa de entrada con activación 'tanh' y regularización L2
    fullyConnectedLayer(256, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros', 'WeightL2Factor', 0.001)
    tanhLayer
    dropoutLayer(0.3)

    % Capa oculta con activación 'tanh' y dropout
    fullyConnectedLayer(128, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
    tanhLayer
    dropoutLayer(0.3)

    % Capa oculta con activación 'tanh' y dropout
    fullyConnectedLayer(128, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
    tanhLayer

    % Capa oculta con activación 'tanh'
    fullyConnectedLayer(64, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
    tanhLayer

    % Capa oculta con activación 'tanh' y dropout
    fullyConnectedLayer(32, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
    tanhLayer
    dropoutLayer(0.3)

    % Capa de salida con 'num_classes' y activación softmax
    fullyConnectedLayer(num_classes, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
    softmaxLayer
    classificationLayer
];

% Configurar opciones de entrenamiento
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_dnn_val, y_dnn_val_cat}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

fprintf('Iniciando el entrenamiento del modelo DNN...\n');
tic_train = tic;
net = trainNetwork(X_dnn_train, y_dnn_train_cat, layers, options);
training_time = toc(tic_train);
fprintf('Entrenamiento del modelo DNN completado.\n');
fprintf('Tiempo de computación del Entrenamiento DNN: %.2f segundos\n', training_time);
fprintf('Entrenamiento completado.\n\n');

fprintf('Iniciando la evaluación del modelo DNN en datos de prueba con ruido específico...\n');

num_noise_levels_eval = 40;
results = struct('snr', cell(1, num_noise_levels_eval), ...
                 'accuracy', cell(1, num_noise_levels_eval), ...
                 'precision', cell(1, num_noise_levels_eval), ...
                 'recall', cell(1, num_noise_levels_eval), ...
                 'f1_score', cell(1, num_noise_levels_eval));
times = zeros(1, num_noise_levels_eval);

noise_levels_db = 40:-1:1;
total_noise_levels = length(noise_levels_db);

progressbar('Evaluando DNN con ruido');

for current_noise_idx = 1:total_noise_levels
    noise_db = noise_levels_db(current_noise_idx);
    start_time_eval = tic; % Iniciar temporizador para cada iteración de evaluación

    % Generar características con ruido específico para el conjunto de prueba
    X_test_current_noise = zeros(length(X_test_clean), num_features);
    for i = 1:length(X_test_clean)
        % preprocess_signal_SNR internamente llama a la preprocess_signal de 35 características
        X_test_current_noise(i, :) = preprocess_signal_SNR(X_test_clean{i}, noise_db, FS);
    end

    % Normalizar usando los scalers ya ajustados del entrenamiento
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
    
    y_test_cat = categorical(y_test); % Convertir etiquetas de prueba a categorical

    % Realizar predicciones
    YPred = classify(net, X_test_normalized);

    % Calcular métricas de rendimiento (similar a GB_Script_Iter.m)
    C_mat = confusionmat(y_test_cat, YPred); % Matriz de confusión

    num_classes_metrics = size(C_mat, 1);
    precision_per_class = zeros(1, num_classes_metrics);
    recall_per_class = zeros(1, num_classes_metrics);
    f1_per_class = zeros(1, num_classes_metrics);
    support_per_class = sum(C_mat, 2); % Suma por filas para obtener el número real de muestras por clase

    for k_metric = 1:num_classes_metrics
        TP = C_mat(k_metric,k_metric);
        FN = sum(C_mat(k_metric,:)) - TP;
        FP = sum(C_mat(:,k_metric)) - TP;

        precision_per_class(k_metric) = TP / (TP + FP);
        recall_per_class(k_metric) = TP / (TP + FN);

        if isnan(precision_per_class(k_metric))
            precision_per_class(k_metric) = 0;
        end
        if isnan(recall_per_class(k_metric))
            recall_per_class(k_metric) = 0;
        end

        f1_per_class(k_metric) = 2 * (precision_per_class(k_metric) * recall_per_class(k_metric)) / (precision_per_class(k_metric) + recall_per_class(k_metric) + eps);
    end

    accuracy = sum(diag(C_mat)) / sum(C_mat(:)); % Precisión general (accuracy)

    % Calcular 'weighted average'
    weighted_precision = sum(precision_per_class .* (support_per_class' / sum(support_per_class)));
    weighted_recall = sum(recall_per_class .* (support_per_class' / sum(support_per_class)));
    weighted_f1_score = sum(f1_per_class .* (support_per_class' / sum(support_per_class)));

    results(current_noise_idx) = struct('snr', noise_db, ...
                            'accuracy', accuracy, ...
                            'precision', weighted_precision, ...
                            'recall', weighted_recall, ...
                            'f1_score', weighted_f1_score);

    fprintf('--- SNR: %d dB | Accuracy: %.4f | F1-Score: %.4f ---\n', noise_db, accuracy, weighted_f1_score);
    fprintf('--- SNR: %d dB | Recall: %.4f | Precision: %.4f ---\n', noise_db, weighted_recall, weighted_precision);
    iteration_time = toc(start_time_eval);
    times(current_noise_idx) = iteration_time;
    fprintf('Tiempo de computación para %d dB: %.2f segundos\n', noise_db, iteration_time);

    progressbar(current_noise_idx / total_noise_levels);
end

% --- GRÁFICOS DE RESULTADOS (CONSISTENTE CON GB_Script_Iter.m) ---
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
ylabel('Métricas de Rendimiento');
title('Rendimiento de la DNN vs. SNR');
legend('Location', 'best');
grid on;
set(gca, 'XDir', 'reverse'); % Invertir eje X (Importante)
hold off;

figure;
plot(snr_values, times, 'g', 'DisplayName', 'Tiempo de Computación');
xlabel('Nivel de Ruido SNR (dB) en el Conjunto de Prueba');
ylabel('Tiempo (segundos)');
title('Tiempo de Computación de la DNN vs. SNR');
legend('Location', 'best');
grid on;
set(gca, 'XDir', 'reverse'); % Invertir eje X (Importante)

fprintf('\nProceso de evaluación completado.\n');