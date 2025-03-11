function [modelParameters] = positionEstimatorTraining(training_data)
    % Arguments:
    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
    
    % Return Value:
    % - modelParameters: structure containing all learned parameters
    
    %% Dimensions of training dataset
    num_angles = size(training_data, 2);
    num_trials = size(training_data, 1);
    num_neurons = size(training_data(1,1).spikes, 1);

    % Model Parameters structure
    modelParameters = struct();
    
%% ================= ANGLE CLASSIFIER (LDA) =================
    fprintf('Training angle classifier (LDA)...\n');
    %% Ranking of Neurons based on Directional Tuning
    avg_fire_rate = zeros(num_neurons, num_angles);

    % The actual movements happen between 300 and 500 ms
    % So we have to isolate firing in this interval
    movement_interval = 300:500;
    
    for angle = 1:num_angles
        firing_rates = [];
    
        for trial = 1:num_trials
            spikes_size = size(training_data(trial,angle).spikes, 2);
            if spikes_size >= max(movement_interval)
                spikes_interval = training_data(trial,angle).spikes(:,movement_interval);
              
                % Calculate the rate of firing in Hz from spike/ms
                firing_rates = [firing_rates, sum(spikes_interval, 2) / length(movement_interval)];
            end
        end 
    
        avg_fire_rate(:, angle) = mean(firing_rates, 2);
    end 

    % Find the tuning depth
    max_firing_rate = max(avg_fire_rate, [], 2);
    min_firing_rate = min(avg_fire_rate, [], 2);
    tune_depth = max_firing_rate - min_firing_rate;

    % Rank the neurons based on the tuning depth
    [~, ranked_neurons] = sort(tune_depth, "descend");
    
    % Select top neurons (top 50%)
    num_top_neurons = ceil(length(ranked_neurons) * 0.5);
    top_neurons = ranked_neurons(1:num_top_neurons);

    %% Feature Extraction (Binning)
    bin_size = 50; 
    
    % Initialize feature and label vectors
    features = []; 
    labels = []; 
    
    % Extract features between 300 and 500 ms (corresponding to movement)
    move_start = 300;
    move_end = 500;
    
    for angle = 1:num_angles
        for trial = 1:num_trials
            % Get spikes within the movement interval
            spike_size = size(training_data(trial, angle).spikes, 2);
            if spike_size >= move_end
                spike_movement = training_data(trial, angle).spikes(top_neurons, move_start:move_end);
        
                num_bins = floor((move_end-move_start+1) / bin_size);
                features_per_trial = zeros(1, length(top_neurons) * num_bins);
                
                % For every neuron bin the data and smooth it
                for neuron = 1:length(top_neurons)
                    for bin = 1:num_bins
                        % Define bin intervals
                        bin_start = (bin-1) * bin_size + 1; 
                        bin_end = min(bin*bin_size, move_end - move_start + 1);
        
                        % Number of spikes in the bin
                        features_per_trial(1, (neuron-1) * num_bins + bin) = sum(spike_movement(neuron, bin_start:bin_end));
                    end 
                end 
        
                % Update main feature matrix with features per trial (Concat)
                features = [features; features_per_trial];
        
                % Update Labels for every angle (Concat)
                labels = [labels; angle];
            end
        end 
    end 
    
    % Normalize the feature vector
    feature_means = mean(features, 1);
    feature_stds = std(features, 0, 1);
    
    % Apply z-score normalization
    features = (features - feature_means) ./ (feature_stds + eps);
    
    fprintf('Extracted %d samples with %d features for classification\n', size(features, 1), size(features, 2));
    
    %% Train LDA Model for Classifying angles from spike data
    % Apply PCA for dimensionality reduction
    use_pca = true;
    
    if use_pca
        % Apply PCA using simplified function
        [coeff, score, latent] = pca(features);
        explained = cumsum(latent) / sum(latent);
        num_components = find(explained >= 0.95, 1, 'first');
        
        features = score(:, 1:num_components);
        
        % Store PCA model
        pca_model = struct();
        pca_model.coeff = coeff(:, 1:num_components);
        pca_model.explained = explained(1:num_components);
        pca_model.means = feature_means;
        pca_model.stds = feature_stds;
    else 
        pca_model = struct();
        pca_model.means = feature_means;
        pca_model.stds = feature_stds;
    end
    
    % Find the mean feature vector for every angle
    uniq_labels = unique(labels);
    num_classes = num_angles;
    centroids = zeros(num_classes, size(features, 2));
    
    for i = 1:num_classes
        class_samples = features(labels == uniq_labels(i), :);
        centroids(i,:) = mean(class_samples, 1);
    end
    
    % Initialize scatter matrices
    Sw = zeros(size(features, 2));  % Within-class scatter (Sw)
    Sb = zeros(size(features, 2));  % Between-class scatter (Sb)
    overall_mean = mean(features, 1);
    
    for i = 1:num_classes
        % Get class samples
        class_samples = features(labels == uniq_labels(i), :);
        class_size = size(class_samples, 1);
        
        % Within-class scatter (Sw)
        class_centered = class_samples - repmat(centroids(i, :), class_size, 1);
        Sw = Sw + (class_centered' * class_centered);
        
        % Between-class scatter (Sb)
        mean_diff = centroids(i, :) - overall_mean;
        Sb = Sb + class_size * (mean_diff' * mean_diff);
    end
    
    % LDA projection: eigenvectors of Sw^-1 * Sb
    [eigvectors, eigvalues] = eig(inv(Sw) * Sb);
    [~, idx] = sort(diag(eigvalues), 'descend');
    eigvectors = eigvectors(:, idx);
    
    % Keep only num_classes-1 eigenvectors
    projection_matrix = eigvectors(:, 1:min(num_classes-1, size(eigvectors, 2)));
    
    % Project centroids
    projected_centroids = centroids * projection_matrix;
    
    % Project features and predict classes for training accuracy
    projected_features = features * projection_matrix;
    predicted_labels = zeros(size(labels));
    
    for i = 1:size(features, 1)
        % Find nearest centroid in projected space
        distances = zeros(num_classes, 1);
        for j = 1:num_classes
            distances(j) = norm(projected_features(i, :) - projected_centroids(j, :));
        end
        [~, min_idx] = min(distances);
        predicted_labels(i) = uniq_labels(min_idx);
    end
    
    % Calculate accuracy
    accuracy = sum(predicted_labels == labels) / length(labels) * 100;
    fprintf('Classifier training accuracy: %.1f%%\n', accuracy);
    
    % Store the LDA model parameters
    LDA_model = struct();
    LDA_model.projection_matrix = projection_matrix;
    LDA_model.centroids = centroids;
    LDA_model.projected_centroids = projected_centroids;
    LDA_model.uniq_labels = uniq_labels;
    LDA_model.pca_model = pca_model;
    LDA_model.top_neurons = top_neurons;
    LDA_model.binwidth = bin_size;
    LDA_model.movement_window = [move_start, move_end];
    
    % Store in model parameters
    modelParameters.classifier = LDA_model;

%% ================= TRAJECTORY REGRESSION =================
    fprintf('Training trajectory regression models...\n');
    
    % Train one regression model per angle for better accuracy
    regression_models = cell(num_angles, 1);
    
    % For each reaching angle
    for angle = 1:num_angles
        % Collect spike counts & corresponding hand positions
        X_data = [];
        xdot_data = [];
        ydot_data = [];

        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes;
            time_bins = size(spikes, 2);
            
            % Skip if trial is too short
            if time_bins < 300
                continue;
            end
            
            % Cumulative spike count - instead of mean firing rate
            cum_spikes = sum(spikes, 2)';
            
            % Target x,y velocities - instead of position
            start_pos = training_data(trial, angle).handPos(1:2, 1);
            final_pos = training_data(trial, angle).handPos(1:2, end);
            duration = size(training_data(trial, angle).handPos, 2);

            v = (final_pos - start_pos) / duration;

            X_data = [X_data; cum_spikes];
            xdot_data = [xdot_data; v(1)];
            ydot_data = [ydot_data; v(2)];
        end
        
        % Skip if not enough data
        if size(X_data, 1) < 5
            % Use default velocity estimate if not enough data
            angle_rad = (angle * 45 - 45) * pi / 180;
            regression_models{angle}.Wx = cos(angle_rad);
            regression_models{angle}.Wy = sin(angle_rad);
            regression_models{angle}.valid = false;
            continue;
        end
        
        % Train model for this angle
        angle_model = struct();
        angle_model.Wx = pinv(X_data) * xdot_data; 
        angle_model.Wy = pinv(X_data) * ydot_data;
        angle_model.valid = true;
        
        regression_models{angle} = angle_model;
    end
    
    % Also train a general model for fallback
    X_data = [];
    xdot_data = [];
    ydot_data = [];

    for angle = 1:num_angles
        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes;
            
            % Skip if trial is too short
            if size(spikes, 2) < 300
                continue;
            end
            
            % Cumulative spike count
            cum_spikes = sum(spikes, 2)';
            
            % Target x,y velocities
            start_pos = training_data(trial, angle).handPos(1:2, 1);
            final_pos = training_data(trial, angle).handPos(1:2, end);
            duration = size(training_data(trial, angle).handPos, 2);

            v = (final_pos - start_pos) / duration;

            X_data = [X_data; cum_spikes];
            xdot_data = [xdot_data; v(1)];
            ydot_data = [ydot_data; v(2)];
        end
    end
    
    % Train general model
    general_model = struct();
    general_model.Wx = pinv(X_data) * xdot_data; 
    general_model.Wy = pinv(X_data) * ydot_data;
    
    % Store in model parameters
    modelParameters.regression = regression_models;
    modelParameters.general_regression = general_model;
    
    fprintf('Model training complete.\n');
end