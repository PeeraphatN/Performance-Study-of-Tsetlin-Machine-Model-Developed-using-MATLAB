function NoisyXOR()
    % Parameters for the Tsetlin Machine
    T = 15; 
    s = 3.9;
    number_of_clauses = 10;
    states = 100; 

    % Parameters of the pattern recognition problem
    number_of_features = 12;
    number_of_classes = 2;

    % Training configuration
    epochs = 200;

    % Loading of training and test data
    training_data = load("C:\Work\Research\Project\DataSet\XOR\Noisy\NoisyXORTrainingData.txt");
    test_data = load("C:\Work\Research\Project\DataSet\XOR\Noisy\NoisyXORTestData.txt");

    X_training = training_data(:, 1:number_of_features); % Input features
    y_training = training_data(:, number_of_features + 1); % Target value

    X_test = test_data(:, 1:number_of_features); % Input features
    y_test = test_data(:, number_of_features + 1); % Target value

    % Create Tsetlin Machine Object
    tsetlin_machine = TsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T);

<<<<<<< HEAD:MATLAB/NoisyXOR.m
    fprintf("Training the Tsetlin Machine on MNIST data ...\n");
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

    starttime = tic;
    tsetlin_machine = tsetlin_machine.fit(X_training, y_training, epochs);
    elapsed_time = toc(starttime);
    fprintf("Training completed. Total time used: %.2f seconds\n", elapsed_time);

    fprintf("\nEvaluating the Tsetlin Machine on test and training data...\n\n");
    acc_test = tsetlin_machine.evaluate(X_test, y_test);
    acc_train = tsetlin_machine.evaluate(X_training, y_training);
    
    fprintf("Accuracy on test data: %.4f\n", acc_test);
    fprintf("Accuracy on training data: %.4f\n", acc_train);
=======
    % Training of the Tsetlin Machine in batch mode
    profile on
    tsetlin_machine = tsetlin_machine.fit(X_training, y_training, epochs);
    profile viewer
    
    % Some performance statistics
    accuracy_test = tsetlin_machine.evaluate(X_test, y_test);
    accuracy_train = tsetlin_machine.evaluate(X_training, y_training);

    fprintf('Accuracy on test data (no noise): %.2f\n', accuracy_test);
    fprintf('Accuracy on training data (40%% noise): %.2f\n', accuracy_train);
>>>>>>> 4f16c792424abaaf25982ec89c7e8ccd3c557706:MATLAB/test_TsetlinMachine.m

    % Predictions
    sample1 = [1,0,1,1,1,0,1,1,1,0,0,0];
    sample2 = [0,1,1,1,1,0,1,1,1,0,0,0];
    sample3 = [0,0,1,1,1,0,1,1,1,0,0,0];
    sample4 = [1,1,1,1,1,0,1,1,1,0,0,0];

    fprintf('Prediction: x1 = 1, x2 = 0 -> y = %d\n', tsetlin_machine.predict(sample1));
    fprintf('Prediction: x1 = 0, x2 = 1 -> y = %d\n', tsetlin_machine.predict(sample2));
    fprintf('Prediction: x1 = 0, x2 = 0 -> y = %d\n', tsetlin_machine.predict(sample3));
    fprintf('Prediction: x1 = 1, x2 = 1 -> y = %d\n', tsetlin_machine.predict(sample4));
end
