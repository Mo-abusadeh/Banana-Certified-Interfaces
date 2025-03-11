function [modelParameters] = linearRegression(data, x_coord, y_coord)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data: string with the name of the file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the data struct
data_struct = load(data);
trial = data_struct.trial; % Extract the trial variable from the loaded struct

% model params. stored in a struct
modelParameters = struct(); % slopes, intercepts, errors

% Get number of each feature
num_neurons = size(trial(1,1).spikes, 1);
num_trials = size(trial, 1);
num_angles = size(trial, 2);

% Temporal Binning Size (Downsampling factor)
bin_size = 10;

% For each angle
for angle = 1:num_angles
    total_features = [];
    total_x_positions = [];
    total_y_positions = [];
    
    % For each trial at current angle
    for t = 1:num_trials % Changed variable name from 'trial' to 't' to avoid confusion
        % spike data for current trial
        spike_data = trial(t,angle).spikes;
        
        % Hand positions for trial (x,y)
        handPos = trial(t,angle).handPos;
        
        % trial length
        trial_length = size(spike_data, 2);
        
        % number of bins
        num_bins = floor(trial_length/bin_size);
        
        % Skip trials that are too short
        if num_bins < 1
            continue;
        end
        
        % Feature vector (binned spikes)
        binned_features = zeros(num_neurons, num_bins);
        
        for bin = 1:num_bins
            % Finds the bin start and end index
            bin_start = (bin-1) * bin_size + 1;
            bin_end = min(bin * bin_size, trial_length);
            
            % Calculate firing rate for each neuron in this bin
            spike_counts = sum(spike_data(:, bin_start:bin_end), 2);
            binned_features(:, bin) = spike_counts / (bin_end - bin_start + 1) * 1000; % Convert to spikes/sec
        end
        
        % resample the hand positions to match the number of bins for
        % the spikes
        resampled_x = zeros(1, num_bins);
        resampled_y = zeros(1, num_bins);
        
        for bin = 1:num_bins
            bin_end = min(bin * bin_size, trial_length);
            resampled_x(bin) = handPos(1, bin_end);
            resampled_y(bin) = handPos(2, bin_end);
        end
        
        % Append data from this trial
        total_features = [total_features, binned_features];
        total_x_positions = [total_x_positions, resampled_x];
        total_y_positions = [total_y_positions, resampled_y];
    end
    
    % Skip angles with no valid trials
    if isempty(total_features)
        fprintf('Warning: No valid trials for angle %d\n', angle);
        continue;
    end
    
    % Linear Regression
    X = total_features';
    
    % Add intercept term
    X = [ones(size(X, 1), 1), X];
    
    % train for x and y separately
    % For X position
    Y_x = total_x_positions'; % Changed from all_x_positions to total_x_positions
    
    % Solve normal equations: beta = (X'X)^(-1)X'Y
    beta_x = (X' * X) \ (X' * Y_x);
    
    % For Y position
    Y_y = total_y_positions'; % Changed from all_y_positions to total_y_positions
    beta_y = (X' * X) \ (X' * Y_y);
    
    % Store coefficients for this angle
    modelParameters.angle(angle).beta_x = beta_x;
    modelParameters.angle(angle).beta_y = beta_y;
end
end