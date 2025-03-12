function predicted_angle = predictAngle(test_data, lda_model, binwidth, top_neurons)
    % Extract spike data from the key window (300-500ms)
    window_start = 300;
    window_end = min(500, size(test_data.spikes, 2));
    
    % If window is too small, return default
    if window_end < window_start + 50
        predicted_angle = 1;
        return;
    end
    
    % Select top neurons
    if window_end >= window_start
        spike_window = test_data.spikes(top_neurons, window_start:window_end);
    else
        predicted_angle = 1;
        return;
    end
    
    % Calculate number of bins - MUST match training exactly
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
    
    % Apply the SAME normalization as in training
    if isfield(lda_model, 'normalization_params')
        % If normalization parameters were stored
        feature_means = lda_model.normalization_params.means;
        feature_stds = lda_model.normalization_params.stds;
        
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
    if isfield(lda_model, 'pca_model') && ~isempty(lda_model.pca_model)
        if isfield(lda_model.pca_model, 'coeff')
            % Apply the PCA transformation
            features = normalized_features * lda_model.pca_model.coeff;
        else
            features = normalized_features;
        end
    else
        features = normalized_features;
    end
    
    % Project features using LDA 
    projected_features = features * lda_model.projection_matrix;
    
    % Find nearest centroid
    distances = zeros(length(lda_model.uniq_labels), 1);
    for j = 1:length(lda_model.uniq_labels)
        distances(j) = norm(projected_features - lda_model.projected_centroids(j, :));
    end
    [~, min_idx] = min(distances);
    
    % Return predicted angle
    predicted_angle = lda_model.uniq_labels(min_idx);
end