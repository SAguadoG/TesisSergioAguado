clear all;
clc;

% Ruta de datos
data_path = 'data_mat'; % Ajustar según sea necesario

% Mapear las etiquetas de clase
label_mapping = struct('flicker_signals', 0, ...
                       'harmonic_signals', 1, ...
                       'interruption_signals', 2, ...
                       'original_signal', 3, ...
                       'sag_signals', 4, ...
                       'swell_signals', 5, ...
                       'transient_signals', 6);

% Inicialización de características y etiquetas
features = [];
labels = [];

% Iterar por cada tipo de perturbación
signal_types = fieldnames(label_mapping);
for i = 1:length(signal_types)
    signal_type = signal_types{i};
    label = label_mapping.(signal_type);
    signal_type_path = fullfile(data_path, signal_type);

    % Verificar si el directorio existe
    if isfolder(signal_type_path)
        subsets = {'train', 'test', 'val'};
        for subset = subsets
            subset_path = fullfile(signal_type_path, subset{1});

            if isfolder(subset_path)
                % Leer archivos en la carpeta
                files = dir(fullfile(subset_path, '*.mat')); % Ajustar según el formato de archivo

                for file = files'
                    % Cargar el archivo .mat
                    loaded_data = load(fullfile(subset_path, file.name));
                    
                    % Obtener el nombre de las variables en el archivo
                    var_names = fieldnames(loaded_data);
                    
                    % Verificar si hay alguna variable en el archivo
                    if ~isempty(var_names)
                        % Supongamos que siempre hay una única variable por archivo, 
                        % accedemos a la primera variable
                        signal = loaded_data.(var_names{1});  % Acceder dinámicamente a la variable
                        
                        % Extraer características
                        feature_vector = preprocess_signal(signal);

                        % Agregar características y etiquetas
                        features = [features; feature_vector];
                        labels = [labels; label];
                    else
                        disp(['Archivo vacío o sin variables: ', file.name]);
                    end
                end
            end
        end
    end
end

% Verificar dimensiones
disp(['Características extraídas: ', num2str(size(features))]);
disp(['Etiquetas extraídas: ', num2str(size(labels))]);

% Dividir los datos en entrenamiento, validación y prueba
rng(42); % Semilla para reproducibilidad
[trainInd, valInd, testInd] = dividerand(size(features, 1), 0.7, 0.15, 0.15);

X_train = features(trainInd, :);
y_train = labels(trainInd);

X_val = features(valInd, :);
y_val = labels(valInd);

X_test = features(testInd, :);
y_test = labels(testInd);

% Convertir etiquetas a one-hot encoding
num_classes = 7;
y_train_one_hot = full(ind2vec(y_train' + 1, num_classes))';
y_val_one_hot = full(ind2vec(y_val' + 1, num_classes))';
y_test_one_hot = full(ind2vec(y_test' + 1, num_classes))';

% Crear el modelo denso (fully connected) similar al de Python
layers = [
    featureInputLayer(size(X_train, 2), 'Normalization', 'zscore')  % Capa de entrada
    
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

    % Capa de salida con 7 clases y activación softmax
    fullyConnectedLayer(num_classes, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
    softmaxLayer
    classificationLayer
];

% Configurar opciones de entrenamiento
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_val, categorical(y_val)}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Entrenar el modelo
net = trainNetwork(X_train, categorical(y_train), layers, options);

% Evaluar el modelo
YPred = classify(net, X_test);
accuracy = mean(YPred == categorical(y_test));
disp(['Test Accuracy: ', num2str(accuracy)]);

% Calcular matriz de confusión
conf_matrix = confusionmat(categorical(y_test), YPred);
disp('Matriz de Confusión:');
disp(conf_matrix);

% Mostrar gráficos de pérdida y precisión
figure;
plotTrainingAccuracyAndLoss(options, net);

% Función para procesar las señales
function feature_vector = preprocess_signal(signal)

    % Media de la señal
    mean_value = mean(signal);

    % Datos sin sesgo
    unbias_data = signal - mean_value;
    unbias_data_2 = unbias_data .^ 2;
    unbias_data_3 = unbias_data_2 .* unbias_data;
    unbias_data_4 = unbias_data_3 .* unbias_data;

    % Cálculo de características
    variance = var(unbias_data);  % Varianza
    skewness = mean(unbias_data_3) / (variance ^ 1.5);  % Asimetría
    kurtosis = mean(unbias_data_4) / (variance ^ 2) - 3;  % Curtosis
    
    % Transformada de Fourier para THD
    fft_signal = fft(signal);
    thd = sqrt(sum(abs(fft_signal(3:4))) / abs(fft_signal(2)));  % Distorsión armónica total
    
    rms = sqrt(mean(signal .^ 2));  % Valor RMS
    crest_factor = max(signal) / rms;  % Factor de cresta

    % Devuelve las características como vector
    feature_vector = [variance, skewness, kurtosis, thd, crest_factor];
end