%% Phase 2: Building the Classifier

load monkeydata_training.mat

% Split into training and validation
[train_data, val_data] = splitDataset(trial, 0.8);

% Dimensions of dataset
num_angles = size(train_data, 2);
num_trials = size(train_data, 1);
num_neurons = size(train_data(1,1).spikes, 1);

%% Ranking Neurons based on Directional Tuning Strength


% Initialise average firing rate for every neuron for every angle
avg_fire_rate = zeros(num_neurons, num_angles);

% The actual movements happen between 300 and 500 ms
% So we have to isolate firing in this interval

movement_interval = 300:500;

for angle = 1:num_angles
    firing_rates = [];

    for trial = 1:size(num_trials)
        
        if size(train_data(trial,angle).spikes, 2) >= max(movement_interval)
            spikes_interval = train_data(trial,angle).spikes(:,movement_interval);
          
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

fprintf("Top 10 neurons ranked based on tuning depth:\n");
for i = 1:10
    fprintf("Neuron #%d: %.2f Hz\n", ranked_neurons(i), tune_depth(ranked_neurons(i)));
end

% Observing the Tuning Strength
figure(1);
bar(tune_depth(ranked_neurons(1:20)));
xlabel("Neuron Number");
ylabel("Tuning Depth (Hz)");
title("Directional Tuning Strength for the Top 20 Neurons");
xticks(1:20);
xticklabels(ranked_neurons(1:20));

% Select top neurons (top 38%)
num_top_neurons = ceil(length(ranked_neurons) * 0.38); % HYPERPARAMETER
top_neurons = ranked_neurons(1:num_top_neurons);


%% Feature Extraction

% The data will be discretised based on a predefined bin size (size of a step)

bin_size = 50; % HYPERPARAMETER: could be changed and tested for 20 or 100 or 200 (50 had the best SNR)

% initialise feature and label vectors
features = []; 
labels = []; 

% Extract features between 300 and 500 ms (corresponding to movement)

move_start = 300;
move_end = 500;

for angle = 1:num_angles
    for trial = 1:num_trials

        % Get spikes within the movement interval
        spike_movement = train_data(trial, angle).spikes(top_neurons, move_start:move_end);

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

% Normalise the feature vector
% Calculate mean and std for each feature
feature_means = mean(features, 1);
feature_stds = std(features, 0, 1);

% Apply z-score normalisation (vectorised)
features = (features - feature_means) ./ (feature_stds + eps);

fprintf("Completed Extracting Features: %d samples with %d features\n", size(features, 1), size(features, 2));

% Store normalisation parameters for prediction
% lda_model.normalization_params.means = feature_means;
% lda_model.normalization_params.stds = feature_stds;

%% Train LDA Model for Classifying angles from spike data


% I want to test with and without using PCA on the feature vector
use_pca = true; % HYPERPARAMETER: change this to false if dont want to use PCA

if use_pca
    [features, pca_model] = applyPCA(features, 0.95);
else 
    pca_model = []
end


% A) Find the mean feature vector for every angle
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

% Keep only num_classes-1 eigenvectors since it is the maximum meaningful
% vectors for LDA
projection_matrix = eigvectors(:, 1:min(num_classes-1, size(eigvectors, 2)));

% Project centroids
projected_centroids = centroids * projection_matrix;

% Project features and predict classes
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

% Store the LDA model
lda_model = struct();
lda_model.projection_matrix = projection_matrix;
lda_model.centroids = centroids;
lda_model.projected_centroids = projected_centroids;
lda_model.uniq_labels = uniq_labels;
lda_model.pca_model = pca_model;

fprintf('Classifier Accuracy on Training Set: %.1f%% \n', accuracy);

% Generate confusion matrix
confusion_mat = zeros(num_classes);
for i = 1:length(labels)
    true_idx = find(uniq_labels == labels(i));
    pred_idx = find(uniq_labels == predicted_labels(i));
    confusion_mat(true_idx, pred_idx) = confusion_mat(true_idx, pred_idx) + 1;
end

% Normalize by row sums 
row_sums = sum(confusion_mat, 2);
for i = 1:num_classes
    if row_sums(i) > 0
        confusion_mat(i, :) = confusion_mat(i, :) / row_sums(i);
    end
end

% Display confusion matrix
figure('Position', [100, 100, 800, 600]);
imagesc(confusion_mat);
colormap('jet');
colorbar;

% Set axis labels
xticks(1:num_classes);
yticks(1:num_classes);
xticklabels(uniq_labels);
yticklabels(uniq_labels);

xlabel('Predicted Angle');
ylabel('True Angle');
title('Confusion Matrix (Normalized)');

% Add text annotations for percentages
for i = 1:num_classes
    for j = 1:num_classes
        text(j, i, sprintf('%.1f%%', confusion_mat(i,j)*100), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', 'w', 'FontWeight', 'bold');
    end
end




%% Validate classifier
fprintf("\nValidating classifier on held-out data...\n");
correct = 0;
total = 0;

num_angles = size(val_data, 2);
num_trials = size(val_data, 1);

for angle = 1:num_angles
    for trial = 1:num_trials
        
        % test data
        test_data = val_data(trial, angle);
        
        % Predict angle
        pred_angle = predictAngle(test_data, lda_model, 50, top_neurons);
        
        % predictions
        if pred_angle == angle
            correct = correct + 1;
        end
        total = total + 1;
    end
end

% Validation accuracy
validation_accuracy = correct / total * 100;
fprintf("Validation accuracy: %.1f%% (%d/%d correct)\n", validation_accuracy, correct, total);

% Save classifier model
classifier_model = struct();
classifier_model.lda_model = lda_model;
classifier_model.top_neurons = top_neurons;
classifier_model.binwidth = 50;
classifier_model.train_accuracy = accuracy;
classifier_model.validation_accuracy = validation_accuracy;

% Save model
save("angle_classifier_model.mat", "classifier_model");
fprintf("Classifier model saved to angle_classifier_model.mat\n");

%% Test our Classification model
