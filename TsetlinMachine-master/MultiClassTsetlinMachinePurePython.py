# multiclass_tsetlin_machine_pure.py
# Pure-Python port matching the logic of the provided Cython implementation.

import numpy as np

class MultiClassTsetlinMachine:
    def __init__(self, number_of_classes, number_of_clauses, number_of_features,
                 number_of_states, s, threshold, boost_true_positive_feedback=0):
        self.number_of_classes = int(number_of_classes)
        self.number_of_clauses = int(number_of_clauses)
        self.number_of_features = int(number_of_features)
        self.number_of_states = int(number_of_states)
        self.s = float(s)
        self.threshold = int(threshold)
        self.boost_true_positive_feedback = int(boost_true_positive_feedback)

        # TA states init: randomly choose either number_of_states or number_of_states+1
        self.ta_state = np.random.choice(
            [self.number_of_states, self.number_of_states + 1],
            size=(self.number_of_clauses, self.number_of_features, 2)
        ).astype(np.int32)

        # Mapping class -> its clauses and their sign
        self.clause_count = np.zeros((self.number_of_classes,), dtype=np.int32)
        self.clause_sign = np.zeros(
            (self.number_of_classes, self.number_of_clauses // self.number_of_classes, 2),
            dtype=np.int32
        )

        # Work buffers
        self.clause_output = np.zeros((self.number_of_clauses,), dtype=np.int32)
        self.class_sum = np.zeros((self.number_of_classes,), dtype=np.int32)
        self.feedback_to_clauses = np.zeros((self.number_of_clauses,), dtype=np.int32)

        # Structure: even clause allocation per class, alternating sign.
        for i in range(self.number_of_classes):
            for j in range(self.number_of_clauses // self.number_of_classes):
                self.clause_sign[i, self.clause_count[i], 0] = i * (self.number_of_clauses // self.number_of_classes) + j
                if j % 2 == 0:
                    self.clause_sign[i, self.clause_count[i], 1] = 1
                else:
                    self.clause_sign[i, self.clause_count[i], 1] = -1
                self.clause_count[i] += 1

    # Translates automata state to action
    def action(self, state: int) -> int:
        if state <= self.number_of_states:
            return 0
        else:
            return 1

    # Calculate the output of each clause using TA actions.
    # predict=1 suppresses all-exclude clauses.
    def calculate_clause_output(self, X: np.ndarray, predict: int = 0):
        X = np.asarray(X, dtype=np.int32, order='C')
        for j in range(self.number_of_clauses):
            self.clause_output[j] = 1
            all_exclude = 1
            for k in range(self.number_of_features):
                action_include = self.action(int(self.ta_state[j, k, 0]))
                action_include_negated = self.action(int(self.ta_state[j, k, 1]))

                if action_include == 1 or action_include_negated == 1:
                    all_exclude = 0

                if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
                    self.clause_output[j] = 0
                    break

            if predict == 1 and all_exclude == 1:
                self.clause_output[j] = 0

    # Sum up the votes for each class
    def sum_up_class_votes(self):
        for target_class in range(self.number_of_classes):
            ssum = 0
            for j in range(int(self.clause_count[target_class])):
                cid = int(self.clause_sign[target_class, j, 0])
                sgn = int(self.clause_sign[target_class, j, 1])
                ssum += int(self.clause_output[cid]) * sgn

            if ssum > self.threshold:
                ssum = self.threshold
            elif ssum < -self.threshold:
                ssum = -self.threshold
            self.class_sum[target_class] = ssum

    ########################################
    ### Predict Target Class for Input X ###
    ########################################

    def predict(self, X: np.ndarray) -> int:
        self.calculate_clause_output(X, predict=1)
        self.sum_up_class_votes()

        max_class_sum = int(self.class_sum[0])
        max_class = 0
        for target_class in range(1, self.number_of_classes):
            if max_class_sum < int(self.class_sum[target_class]):
                max_class_sum = int(self.class_sum[target_class])
                max_class = target_class
        return max_class

    # Get the state of a specific automaton
    def get_state(self, clause: int, feature: int, automaton_type: int) -> int:
        return int(self.ta_state[clause, feature, automaton_type])

    ############################################
    ### Evaluate the Trained Tsetlin Machine ###
    ############################################

    def evaluate(self, X: np.ndarray, y: np.ndarray, number_of_examples: int) -> float:
        Xi = np.zeros((self.number_of_features,), dtype=np.int32)
        errors = 0

        for l in range(number_of_examples):
            # Copy row into Xi (to match original per-sample buffer behavior)
            for j in range(self.number_of_features):
                Xi[j] = int(X[l, j])

            self.calculate_clause_output(Xi, predict=1)
            self.sum_up_class_votes()

            max_class_sum = int(self.class_sum[0])
            max_class = 0
            for target_class in range(1, self.number_of_classes):
                if max_class_sum < int(self.class_sum[target_class]):
                    max_class_sum = int(self.class_sum[target_class])
                    max_class = target_class

            if max_class != int(y[l]):
                errors += 1

        return 1.0 - 1.0 * errors / float(number_of_examples)

    ##########################################
    ### Online Training of Tsetlin Machine ###
    ##########################################

    def update(self, X: np.ndarray, target_class: int):
        # Randomly pick one of the other classes
        negative_target_class = np.random.randint(0, self.number_of_classes)
        while negative_target_class == target_class:
            negative_target_class = np.random.randint(0, self.number_of_classes)

        # Calculate clause output and class votes
        self.calculate_clause_output(X, predict=0)
        self.sum_up_class_votes()

        # Initialize feedback
        self.feedback_to_clauses.fill(0)

        # Feedback to clauses for target_class
        for j in range(int(self.clause_count[target_class])):
            # if 1.0*rand()/RAND_MAX > (1.0/(threshold*2))*(threshold - class_sum[target]) : continue
            if np.random.random() > (1.0 / (self.threshold * 2.0)) * (self.threshold - float(self.class_sum[target_class])):
                continue

            if int(self.clause_sign[target_class, j, 1]) >= 0:
                self.feedback_to_clauses[int(self.clause_sign[target_class, j, 0])] = 1   # Type I
            else:
                self.feedback_to_clauses[int(self.clause_sign[target_class, j, 0])] = -1  # Type II

        # Feedback to clauses for negative_target_class
        for j in range(int(self.clause_count[negative_target_class])):
            # if 1.0*rand()/RAND_MAX > (1.0/(threshold*2))*(threshold + class_sum[neg]) : continue
            if np.random.random() > (1.0 / (self.threshold * 2.0)) * (self.threshold + float(self.class_sum[negative_target_class])):
                continue

            if int(self.clause_sign[negative_target_class, j, 1]) >= 0:
                self.feedback_to_clauses[int(self.clause_sign[negative_target_class, j, 0])] = -1  # Type II
            else:
                self.feedback_to_clauses[int(self.clause_sign[negative_target_class, j, 0])] = 1   # Type I

        # Train individual automata
        for j in range(self.number_of_clauses):
            fb = int(self.feedback_to_clauses[j])
            if fb > 0:
                # Type I Feedback (Combats False Negatives)
                if int(self.clause_output[j]) == 0:
                    for k in range(self.number_of_features):
                        if np.random.random() <= 1.0 / self.s:
                            if int(self.ta_state[j, k, 0]) > 1:
                                self.ta_state[j, k, 0] -= 1
                        if np.random.random() <= 1.0 / self.s:
                            if int(self.ta_state[j, k, 1]) > 1:
                                self.ta_state[j, k, 1] -= 1

                elif int(self.clause_output[j]) == 1:
                    for k in range(self.number_of_features):
                        if int(X[k]) == 1:
                            if self.boost_true_positive_feedback == 1 or np.random.random() <= (self.s - 1.0) / self.s:
                                if int(self.ta_state[j, k, 0]) < self.number_of_states * 2:
                                    self.ta_state[j, k, 0] += 1
                            if np.random.random() <= 1.0 / self.s:
                                if int(self.ta_state[j, k, 1]) > 1:
                                    self.ta_state[j, k, 1] -= 1

                        elif int(X[k]) == 0:
                            if self.boost_true_positive_feedback == 1 or np.random.random() <= (self.s - 1.0) / self.s:
                                if int(self.ta_state[j, k, 1]) < self.number_of_states * 2:
                                    self.ta_state[j, k, 1] += 1
                            if np.random.random() <= 1.0 / self.s:
                                if int(self.ta_state[j, k, 0]) > 1:
                                    self.ta_state[j, k, 0] -= 1

            elif fb < 0:
                # Type II Feedback (Combats False Positives)
                if int(self.clause_output[j]) == 1:
                    for k in range(self.number_of_features):
                        action_include = self.action(int(self.ta_state[j, k, 0]))
                        action_include_negated = self.action(int(self.ta_state[j, k, 1]))

                        if int(X[k]) == 0:
                            if action_include == 0 and int(self.ta_state[j, k, 0]) < self.number_of_states * 2:
                                self.ta_state[j, k, 0] += 1
                        elif int(X[k]) == 1:
                            if action_include_negated == 0 and int(self.ta_state[j, k, 1]) < self.number_of_states * 2:
                                self.ta_state[j, k, 1] += 1

    ##############################################
    ### Batch Mode Training of Tsetlin Machine ###
    ##############################################

    def fit(self, X: np.ndarray, y: np.ndarray, number_of_examples: int, epochs: int = 100):
        Xi = np.zeros((self.number_of_features,), dtype=np.int32)
        random_index = np.arange(number_of_examples, dtype=np.int32)

        acc_log = []

        for epoch in range(epochs):
            np.random.shuffle(random_index)

            for i in range(number_of_examples):
                example_id = int(random_index[i])
                target_class = int(y[example_id])

                for j in range(self.number_of_features):
                    Xi[j] = int(X[example_id, j])

                self.update(Xi, target_class)

            acc = self.evaluate(X, y, number_of_examples)
            acc_log.append(acc)

        return acc_log

    # Kept the original misspelled method name to match the source
    def print_caluse_signs(self):
        for i in range(self.number_of_classes):
            print("Class", i, "Clauses:")
            for j in range(int(self.clause_count[i])):
                print("Clause", int(self.clause_sign[i, j, 0]), "Sign:", int(self.clause_sign[i, j, 1]))
            print()
