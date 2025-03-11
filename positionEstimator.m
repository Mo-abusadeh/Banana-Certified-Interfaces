function [x, y] = positionEstimator(test_data, modelParameters)
    % This function predicts the hand position (x, y) using a two-stage approach:
    % 1. Classify the reaching angle using LDA
    % 2. Use the regression model specific to that angle to predict trajectory
    
    % For the first call, initialize previous prediction
    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
        first_prediction = true;
    else
        % Get the last predicted position
        prev_x = test_data.decodedHandPos(1, end);
        prev_y = test_data.decodedHandPos(2, end);
        first_prediction = false;
    end
    
    % Current timestamp
    current_time = size(test_data.spikes, 2);
    
    % Stage 1: Classify the reaching angle
    % Extract spike data from the key window (300-500ms)
    window_start = modelParameters.classifier.movement_window(1);
    window_end = min(modelParameters.classifier.movement_window(2), current_time);
    
    % If we don't have enough data yet, use simple trajectory prediction
    if current_time < window_start + 50
        % Early in the trial, just predict a small movement
        % Use the general regression model for initial estimate
        spikes = test_data.spikes;
        cum_spikes = sum(spikes, 2)';
        
        xdot = cum_spikes * modelParameters.general_regression.Wx;
        ydot = cum_spikes * modelParameters.general_regression.Wy;
        
        % Make a small step from previous position
        step_size = 20; % ms
        x = prev_x + xdot * step_size/1000;
        y = prev_y + ydot * step_size/1000;
        return;
    end
    
    % Select top neurons
    top_neurons = modelParameters.classifier.top_neurons;
    spike_window = test_data.spikes(top_neurons, window_start:window_end);
    
    % Calculate number of bins - MUST match training exactly
    binwidth = modelParameters.classifier.binwidth;
    num_bins = floor((window_end - window_start + 1) / binwidth);
    
    % Extract features - exactly as in training
    features_per_trial = zeros(1, length(top_neurons) * num_bins);
    
    for neuron = 1:length(top_neurons)
        for bin = 1:num_bins
            % Define bin intervals
            bin_start = (bin-1) * binwidth + 1; 
            bin_end = min(bin*binwidth, window_end - window_start + 1);
            
            % Exactly match training feature extraction
            features_per_trial(1, (neuron-1) * num_bins + bin) = sum(spike_window(neuron, bin_start:bin_end));
        end
    end
    
    % Apply the same normalization as in training
    if isfield(modelParameters.classifier.pca_model, 'means')
        feature_means = modelParameters.classifier.pca_model.means;
        feature_stds = modelParameters.classifier.pca_model.stds;
        
        % Ensure dimensions match
        if length(features_per_trial) < length(feature_means)
            % Pad with zeros if needed
            features_per_trial = [features_per_trial, zeros(1, length(feature_means) - length(features_per_trial))];
        elseif length(features_per_trial) > length(feature_means)
            % Truncate if too long
            features_per_trial = features_per_trial(1:length(feature_means));
        end
        
        % Apply normalization
        normalized_features = (features_per_trial - feature_means) ./ (feature_stds + eps);
    else
        % If no stored params, use local normalization (less ideal)
        normalized_features = (features_per_trial - mean(features_per_trial)) ./ (std(features_per_trial) + eps);
    end
    
    % Apply PCA if it was used in training
    if isfield(modelParameters.classifier.pca_model, 'coeff') && ~isempty(modelParameters.classifier.pca_model.coeff)
        % Apply the PCA transformation
        features = normalized_features * modelParameters.classifier.pca_model.coeff;
    else
        features = normalized_features;
    end
    
    % Project features using LDA 
    projected_features = features * modelParameters.classifier.projection_matrix;
    
    % Find nearest centroid
    distances = zeros(length(modelParameters.classifier.uniq_labels), 1);
    for j = 1:length(modelParameters.classifier.uniq_labels)
        distances(j) = norm(projected_features - modelParameters.classifier.projected_centroids(j, :));
    end
    [~, min_idx] = min(distances);
    
    % Predicted angle
    predicted_angle = modelParameters.classifier.uniq_labels(min_idx);
    
    % Stage 2: Use the regression model for the predicted angle
    % Get spike data
    spikes = test_data.spikes;
    cum_spikes = sum(spikes, 2)';
    
    % Use angle-specific regression model if valid, otherwise fallback to general model
    if modelParameters.regression{predicted_angle}.valid
        reg_model = modelParameters.regression{predicted_angle};
    else
        reg_model = modelParameters.general_regression;
    end
    
    % Predict velocity
    xdot = cum_spikes * reg_model.Wx;
    ydot = cum_spikes * reg_model.Wy;
    
    % Calculate time step (ms)
    if first_prediction
        % First prediction, use full time
        time_step = current_time;
    else
        % Subsequent predictions, use increment
        time_step = 20; % From the test script, it's always 20ms
    end
    
    % Euler integration - predicted position
    x = prev_x + xdot * time_step/1000; % Convert ms to seconds
    y = prev_y + ydot * time_step/1000;
    
    % Add some momentum-based smoothing for more stable predictions
    % This can be adjusted or removed if it doesn't improve results
    if ~first_prediction && false  % Currently disabled with 'false'
        smoothing_factor = 0.7;  % Adjust between 0-1
        x = smoothing_factor * x + (1-smoothing_factor) * prev_x;
        y = smoothing_factor * y + (1-smoothing_factor) * prev_y;
    end
end