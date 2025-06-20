classdef TsetlinMachine < handle
    properties
        number_of_classes
        number_of_clauses
        number_of_features
        s
        number_of_states
        threshold
        boost_true_positive_feedback
        ta_state
        clause_count
        clause_sign
        clause_output
        class_sum
        feedback_to_clauses
    end

    methods
        function obj = TsetlinMachine(number_of_classes, number_of_clauses, number_of_features, number_of_states, s, threshold, boost_true_positive_feedback)
            if nargin < 7
                boost_true_positive_feedback = 0;
            end

            obj.number_of_classes = number_of_classes;
            obj.number_of_clauses = number_of_clauses;
            obj.number_of_features = number_of_features;
            obj.number_of_states = number_of_states;
            obj.s = s;
            obj.threshold = threshold;
            obj.boost_true_positive_feedback = boost_true_positive_feedback;

            % Initialize TA state
            obj.ta_state = randi([number_of_states, number_of_states + 1], number_of_clauses, number_of_features, 2);

            % Initialize Clause Sign and Count
            obj.clause_count = zeros(number_of_classes, 1);
            obj.clause_sign = zeros(number_of_classes, floor(number_of_clauses / number_of_classes), 2);
            obj.clause_output = zeros(number_of_clauses, 1);
            obj.class_sum = zeros(number_of_classes, 1);
            obj.feedback_to_clauses = zeros(number_of_clauses, 1);

            % Set up the Tsetlin Machine structure
            for i = 1:number_of_classes
                for j = 1:floor(number_of_clauses / number_of_classes)
                    obj.clause_sign(i, obj.clause_count(i) + 1, 1) = (i - 1) * floor(number_of_clauses / number_of_classes) + j;
                    if mod(j, 2) == 1
                        obj.clause_sign(i, obj.clause_count(i) + 1, 2) = 1;
                    else
                        obj.clause_sign(i, obj.clause_count(i) + 1, 2) = -1;
                    end
                    obj.clause_count(i) = obj.clause_count(i) + 1;
                end
            end
        end

        function obj = calculate_clause_output(obj, X, predict)
            if nargin < 3
                predict = 0;
            end
        
            ta_inc_pos = obj.ta_state(:,:,1) > obj.number_of_states;
            ta_inc_neg = obj.ta_state(:,:,2) > obj.number_of_states;
        
            X_mat = repmat(X, obj.number_of_clauses, 1); % กระจาย input
        
            fail_inc = ta_inc_pos & (X_mat == 0);
            fail_neg = ta_inc_neg & (X_mat == 1);
        
            clause_fail = any(fail_inc | fail_neg, 2);
            obj.clause_output = ~clause_fail;
        
            if predict
                all_exclude = ~any(ta_inc_pos | ta_inc_neg, 2);
                obj.clause_output(all_exclude) = 0;
            end
        end

        function obj = sum_up_class_votes(obj)
            obj.class_sum = zeros(obj.number_of_classes, 1);

            for target_class = 1:obj.number_of_classes
                for j = 1:obj.clause_count(target_class)
                    clause_index = obj.clause_sign(target_class, j, 1);
                    obj.class_sum(target_class) = obj.class_sum(target_class) + obj.clause_output(clause_index) * obj.clause_sign(target_class, j, 2);
                end

                obj.class_sum(target_class) = min(max(obj.class_sum(target_class), -obj.threshold), obj.threshold);
            end
        end

        function predicted_class = predict(obj, X)
            obj = obj.calculate_clause_output(X, 1);
            obj = obj.sum_up_class_votes();
            [~, predicted_class] = max(obj.class_sum);
            predicted_class = predicted_class - 1;
        end

    function accuracy = evaluate(obj, X, y)
        num_samples = size(X, 1);
        correct_predictions = 0;

        for i = 1:num_samples
            predicted_class = obj.predict(X(i, :));
            if predicted_class == y(i)         
                correct_predictions = correct_predictions + 1;
            end
        end
        accuracy = correct_predictions / num_samples;
    end

    function obj = update(obj, X, target_class)
        classes = 1:obj.number_of_classes;
        classes(target_class) = [];
        negative_target_class = classes(randi(length(classes)));
    
        obj = obj.calculate_clause_output(X, 0);
        obj = obj.sum_up_class_votes();
    
        obj.feedback_to_clauses = zeros(obj.number_of_clauses, 1);
    
        clause_list  = reshape(obj.clause_sign(target_class,:,1), [], 1);
        clause_signs = reshape(obj.clause_sign(target_class,:,2), [], 1);
        num_clauses     = length(clause_list);
        rand_vals       = rand(num_clauses, 1);
        prob_pos        = (1 / (obj.threshold * 2)) * (obj.threshold - obj.class_sum(target_class));
        selected_pos    = rand_vals <= prob_pos;
        
        obj.feedback_to_clauses(clause_list(selected_pos)) = clause_signs(selected_pos);
    
        clause_list  = reshape(obj.clause_sign(negative_target_class,:,1), [], 1);
        clause_signs = reshape(obj.clause_sign(negative_target_class,:,2), [], 1);
        num_clauses     = length(clause_list);
        rand_vals       = rand(num_clauses, 1);
        prob_neg        = (1 / (obj.threshold * 2)) * (obj.threshold + obj.class_sum(negative_target_class));
        selected_neg    = rand_vals <= prob_neg;
        
        obj.feedback_to_clauses(clause_list(selected_neg)) = -clause_signs(selected_neg);
    
        X_mat = repmat(X, obj.number_of_clauses, 1);  % broadcast

        idx_pos_zero = obj.feedback_to_clauses > 0 & obj.clause_output == 0;
        idx_pos_one  = obj.feedback_to_clauses > 0 & obj.clause_output == 1;
        idx_neg      = obj.feedback_to_clauses < 0 & obj.clause_output == 1;

        if any(idx_pos_zero)
            rand_mask = rand(sum(idx_pos_zero), obj.number_of_features) <= 1.0 / obj.s;
        
            ta1 = obj.ta_state(idx_pos_zero, :, 1);
            ta2 = obj.ta_state(idx_pos_zero, :, 2);
        
            ta1(rand_mask & ta1 > 1) = ta1(rand_mask & ta1 > 1) - 1;
            ta2(rand_mask & ta2 > 1) = ta2(rand_mask & ta2 > 1) - 1;
        
            obj.ta_state(idx_pos_zero, :, 1) = ta1;
            obj.ta_state(idx_pos_zero, :, 2) = ta2;
        end
        if any(idx_pos_one)
            ta1 = obj.ta_state(idx_pos_one, :, 1);
            ta2 = obj.ta_state(idx_pos_one, :, 2);
        
            x_eq1 = X_mat(idx_pos_one, :) == 1;
            x_eq0 = ~x_eq1;
        
            rand1 = rand(sum(idx_pos_one), obj.number_of_features);
            rand2 = rand(sum(idx_pos_one), obj.number_of_features);
        
            boost_mask = obj.boost_true_positive_feedback == 1 | (rand1 <= (obj.s - 1)/obj.s);
        
            % X == 1 → เพิ่ม include
            inc1 = x_eq1 & boost_mask & (ta1 < obj.number_of_states * 2);
            ta1(inc1) = ta1(inc1) + 1;
        
            % X == 1 → ลด include_negated
            dec2 = x_eq1 & (rand2 <= 1.0/obj.s) & (ta2 > 1);
            ta2(dec2) = ta2(dec2) - 1;
        
            % X == 0 → เพิ่ม include_negated
            inc2 = x_eq0 & boost_mask & (ta2 < obj.number_of_states * 2);
            ta2(inc2) = ta2(inc2) + 1;
        
            % X == 0 → ลด include
            dec1 = x_eq0 & (rand2 <= 1.0/obj.s) & (ta1 > 1);
            ta1(dec1) = ta1(dec1) - 1;
        
            obj.ta_state(idx_pos_one, :, 1) = ta1;
            obj.ta_state(idx_pos_one, :, 2) = ta2;
        end
        if any(idx_neg)
            ta1 = obj.ta_state(idx_neg, :, 1);
            ta2 = obj.ta_state(idx_neg, :, 2);
        
            inc_mask1 = (X_mat(idx_neg,:) == 0) & (ta1 <= obj.number_of_states);
            inc_mask2 = (X_mat(idx_neg,:) == 1) & (ta2 <= obj.number_of_states);
        
            ta1(inc_mask1 & ta1 < obj.number_of_states * 2) = ta1(inc_mask1 & ta1 < obj.number_of_states * 2) + 1;
            ta2(inc_mask2 & ta2 < obj.number_of_states * 2) = ta2(inc_mask2 & ta2 < obj.number_of_states * 2) + 1;
        
            obj.ta_state(idx_neg, :, 1) = ta1;
            obj.ta_state(idx_neg, :, 2) = ta2;
        end

    end

    function obj = update_parallel(obj, X, target_class)
        % Precompute
        obj = obj.calculate_clause_output(X, 0);
        obj = obj.sum_up_class_votes();
    
        % Prep
        num_clauses = obj.number_of_clauses;
        num_feat = obj.number_of_features;
        num_states = obj.number_of_states;
        max_state = 2 * num_states;
    
        ta_state = obj.ta_state; % copy
        feedback = zeros(num_clauses, 1);
    
        % Negative class
        classes = 1:obj.number_of_classes;
        classes(target_class) = [];
        neg_class = classes(randi(length(classes)));
    
        % Broadcast
        clause_sign = obj.clause_sign;
        clause_output = obj.clause_output;
        class_sum = obj.class_sum;
        boost = obj.boost_true_positive_feedback;
        X_mat = repmat(X, num_clauses, 1);
    
        % Parallel per clause chunk
        parfor j = 1:num_clauses
            % 1. หาว่า clause นี้อยู่ class ไหน
            class = -1;
            sign = 0;
            for c = 1:obj.number_of_classes
                idx = clause_sign(c, :, 1);
                idx = idx(:);
                found = find(idx == j, 1);
                if ~isempty(found)
                    class = c;
                    sign = clause_sign(c, found, 2);
                    break;
                end
            end
    
            % 2. คิด feedback
            if class == target_class
                p = (1 / (obj.threshold * 2)) * (obj.threshold - class_sum(class));
                if rand <= p
                    feedback(j) = sign;
                end
            elseif class == neg_class
                p = (1 / (obj.threshold * 2)) * (obj.threshold + class_sum(class));
                if rand <= p
                    feedback(j) = -sign;
                end
            end
        end
    
        % 3. Feedback → vectorized update
        idx_pos0 = feedback > 0 & obj.clause_output == 0;
        idx_pos1 = feedback > 0 & obj.clause_output == 1;
        idx_neg  = feedback < 0 & obj.clause_output == 1;
    
        % -- POS0: clause output = 0 → penalize
        if any(idx_pos0)
            mask = rand(sum(idx_pos0), num_feat) <= 1.0 / obj.s;
            ta = ta_state(idx_pos0, :, :);
            ta1 = ta(:, :, 1); ta2 = ta(:, :, 2);
            ta1(mask & ta1 > 1) = ta1(mask & ta1 > 1) - 1;
            ta2(mask & ta2 > 1) = ta2(mask & ta2 > 1) - 1;
            ta(:, :, 1) = ta1; ta(:, :, 2) = ta2;
            ta_state(idx_pos0, :, :) = ta;
        end
    
        % -- POS1: clause output = 1 → reinforce
        if any(idx_pos1)
            ta = ta_state(idx_pos1, :, :);
            ta1 = ta(:, :, 1); ta2 = ta(:, :, 2);
            Xb = X_mat(idx_pos1, :);
            rand1 = rand(sum(idx_pos1), num_feat);
            rand2 = rand(sum(idx_pos1), num_feat);
            boost_mask = boost == 1 | rand1 <= (obj.s - 1) / obj.s;
    
            % X==1: inc include, dec include_negated
            inc1 = Xb == 1 & boost_mask & ta1 < max_state;
            dec2 = Xb == 1 & rand2 <= 1.0 / obj.s & ta2 > 1;
            ta1(inc1) = ta1(inc1) + 1;
            ta2(dec2) = ta2(dec2) - 1;
    
            % X==0: inc include_negated, dec include
            inc2 = Xb == 0 & boost_mask & ta2 < max_state;
            dec1 = Xb == 0 & rand2 <= 1.0 / obj.s & ta1 > 1;
            ta2(inc2) = ta2(inc2) + 1;
            ta1(dec1) = ta1(dec1) - 1;
    
            ta(:, :, 1) = ta1; ta(:, :, 2) = ta2;
            ta_state(idx_pos1, :, :) = ta;
        end
    
        % -- NEG: clause output = 1 → suppress wrong class
        if any(idx_neg)
            ta = ta_state(idx_neg, :, :);
            ta1 = ta(:, :, 1); ta2 = ta(:, :, 2);
            Xb = X_mat(idx_neg, :);
            mask1 = Xb == 0 & ta1 <= num_states & ta1 < max_state;
            mask2 = Xb == 1 & ta2 <= num_states & ta2 < max_state;
            ta1(mask1) = ta1(mask1) + 1;
            ta2(mask2) = ta2(mask2) + 1;
            ta(:, :, 1) = ta1; ta(:, :, 2) = ta2;
            ta_state(idx_neg, :, :) = ta;
        end
    
        % Final assign
        obj.ta_state = ta_state;
        obj.feedback_to_clauses = feedback;
    end


    function obj = fit(obj, X, y, epochs)
        for epoch = 1:epochs
            for i = 1:size(X, 1)
                target_class = y(i) + 1;
                if target_class > obj.number_of_classes
                    error('target_class=%d exceeds number_of_classes=%d', target_class, obj.number_of_classes);
                end
                obj = obj.update(X(i, :), target_class);
            end
        end
    end

    end
end
