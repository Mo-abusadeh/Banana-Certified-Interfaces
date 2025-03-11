function [trainData, validData] = splitDataset(trial_data, train_ratio)
    % Splits the trial data into training and validation sets
    %
    % Inputs:
    %   trial_data - Full training dataset (100x8 struct array)
    %   train_ratio - Proportion of data to use for training (e.g., 0.8)
    %
    % Outputs:
    %   trainData - Training dataset
    %   validData - Validation dataset
    
    if nargin < 2
        train_ratio = 0.8;
    end
    
    num_trials = size(trial_data, 1);
    num_angles = size(trial_data, 2);
    
    % Calculate split indices
    num_train_trials = round(num_trials * train_ratio);
    
    % Randomly shuffle trial indices
    rng(42); % For reproducibility
    shuffled_indices = randperm(num_trials);
    
    train_indices = shuffled_indices(1:num_train_trials);
    valid_indices = shuffled_indices(num_train_trials+1:end);
    
    % Create training and validation datasets
    trainData = trial_data(train_indices, :);
    validData = trial_data(valid_indices, :);
    
    % Display info about the split
    fprintf('Dataset split complete:\n');
    fprintf(' - Training set: %d trials per angle\n', num_train_trials);
    fprintf(' - Validation set: %d trials per angle\n', num_trials - num_train_trials);
end