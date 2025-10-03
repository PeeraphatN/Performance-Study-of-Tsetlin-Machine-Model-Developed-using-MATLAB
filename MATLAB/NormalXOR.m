function NormalXOR(varargin)
    %% ---------- PID Detection ----------
    pid = feature('getpid'); 
    fid = fopen('C:\temp\training_pid.txt', 'w'); 
    if fid == -1 
        error('Cannot open C:\temp\training_pid.txt to write.'); 
    end 
    fprintf(fid, '%d\n', pid); 
    fclose(fid);
    %% -----------------------------------

    %% --------- Parse CLI arguments ---------
    p = inputParser;
    addParameter(p, 'T', 15, @isnumeric);
    addParameter(p, 's', 3.9, @isnumeric);
    addParameter(p, 'clauses', 10, @isnumeric);
    addParameter(p, 'states', 100, @isnumeric);
    addParameter(p, 'epochs', 200, @isnumeric);
    parse(p, varargin{:});

    T = p.Results.T;
    s = p.Results.s;
    number_of_clauses = p.Results.clauses;
    states = p.Results.states;
    epochs = p.Results.epochs;
    %% ---------------------------------------

    %% Problem configuration
    number_of_features = 2;
    number_of_classes = 2;

    %% Load dataset
    training_data = load("..\DataSet\XOR\Normal\NormalXORTrainingData.txt");
    test_data = load("..\DataSet\XOR\Normal\NormalXORTestData.txt");

    X_training = training_data(:, 1:number_of_features);
    y_training = training_data(:, number_of_features + 1);

    X_test = test_data(:, 1:number_of_features);
    y_test = test_data(:, number_of_features + 1);

    %% Create TM object
    tsetlin_machine = TsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T);

    %% Print configuration
    fprintf("Training the Tsetlin Machine on Normal XOR data ...\n");
    fprintf("Hyperparameters:\n");
    fprintf("Number of features: %d\n", number_of_features);
    fprintf("Number of classes: %d\n", number_of_classes);
    fprintf("T: %d\n", T);
    fprintf("s: %.2f\n", s);
    fprintf("Number of clauses: %d\n", number_of_clauses);
    fprintf("Number of states: %d\n", states);
    fprintf("Epochs: %d\n", epochs);
    fprintf("Number of training samples: %d\n", length(y_training));
    fprintf("Number of test samples: %d\n", length(y_test));

    %% Training
    starttime = tic;
    [tsetlin_machine, acc_log] = tsetlin_machine.fit(X_training, y_training, epochs);
    elapsed_time = toc(starttime);

    fprintf("Training completed. Total time used: %.2f seconds\n", elapsed_time);

    %% Evaluation
    acc_test = tsetlin_machine.evaluate(X_test, y_test);
    acc_train = tsetlin_machine.evaluate(X_training, y_training);

    fprintf("\nEvaluating the Tsetlin Machine on test and training data...\n");
    fprintf("Accuracy on test data: %.4f\n", acc_test);
    fprintf("Accuracy on training data: %.4f\n", acc_train);

    %% Prediction samples
    sample1 = [1,0];
    sample2 = [0,1];
    sample3 = [0,0];
    sample4 = [1,1];

    fprintf('Prediction: x1 = 1, x2 = 0 -> y = %d\n', tsetlin_machine.predict(sample1));
    fprintf('Prediction: x1 = 0, x2 = 1 -> y = %d\n', tsetlin_machine.predict(sample2));
    fprintf('Prediction: x1 = 0, x2 = 0 -> y = %d\n', tsetlin_machine.predict(sample3));
    fprintf('Prediction: x1 = 1, x2 = 1 -> y = %d\n', tsetlin_machine.predict(sample4));

    %% Create result folder
    log_dir = fullfile("MATLAB", "result", "normal_xor");
    if ~exist(log_dir, 'dir')
        mkdir(log_dir);
    end

    %% Save epoch-wise accuracy log
    timestamp = datestr(now, "yyyymmdd_HHMMSS");
    epoch_log_path = fullfile(log_dir, sprintf("normal_xor_epoch_log_%s.csv", timestamp));

    acc_log = acc_log(:);
    epochs_col = (1:epochs)';
    log_table = table(epochs_col, acc_log, 'VariableNames', {'epoch', 'accuracy'});
    header_lines = {
        sprintf("%% T: %d", T);
        sprintf("%% s: %.2f", s);
        sprintf("%% clauses: %d", number_of_clauses);
        sprintf("%% states: %d", states);
        sprintf("%% epochs: %d", epochs);
        sprintf("%% time: %.4f\n", elapsed_time);

        sprintf("%% acc_test: %.4f", acc_test);
        sprintf("%% acc_train: %.4f", acc_train);

    };
    fid = fopen(epoch_log_path, 'w');
    for i = 1:length(header_lines)
        fprintf(fid, '%s\n', header_lines{i});
    end
    fclose(fid);

    writetable(log_table, epoch_log_path, 'WriteMode', 'Append');

    fprintf("Epoch-wise accuracy log saved to %s\n", epoch_log_path);

    %% Save summary log (append mode)
    summary_path = fullfile(log_dir, "normalXOR_result_log.csv");
    file_exists = exist(summary_path, 'file');

    summary_data = table(number_of_features, number_of_classes, T, s, number_of_clauses, ...
        states, epochs, acc_test, acc_train, elapsed_time, ...
        'VariableNames', {'number_of_features', 'number_of_classes', 'T', 's', ...
        'number_of_clauses', 'number_of_states', 'epochs', ...
        'Accuracy_on_test_data', 'Accuracy_on_training_data', 'Time'});

    if file_exists
        writetable(summary_data, summary_path, 'WriteMode', 'Append');
    else
        writetable(summary_data, summary_path);
    end

    %% ---------- Clean up PID ----------
    pause(1);
    if exist('C:\temp\training_pid.txt', 'file')
        delete('C:\temp\training_pid.txt');
        fprintf('PID file deleted.\n');
    end
    %% ----------------------------------
end
