function MNIST()
    T = 15; 
    s = 3.9;
    number_of_clauses = 1000;
    states = 100; 

    number_of_features = 784;
    number_of_classes = 10;
    epochs = 500;

    training_data = load("C:\Work\Research\Project\DataSet\MNIST\MNISTTraining.txt");
    test_data = load("C:\Work\Research\Project\DataSet\MNIST\MNISTTest.txt");

    X_training = training_data(:, 1:number_of_features); 
    y_training = training_data(:, number_of_features + 1); 

    X_test = test_data(:, 1:number_of_features); 
    y_test = test_data(:, number_of_features + 1); 

    tsetlin_machine = TsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T);

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

    fprintf("\nPredictions for first 20 test samples:\n");
    for i = 1:20
        pred = tsetlin_machine.predict(X_test(i, :));
        fprintf("Sample %d: Predicted class = %d, True class = %d\n", i, pred, y_test(i));
    end
end
