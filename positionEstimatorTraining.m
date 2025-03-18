function [modelParameters] = positionEstimatorTraining(training_data)
    % Arguments:
    % - training_data: A 2D matrix of struct data with trials and angles
    %   - training_data(n,k).trialId      unique trial ID
    %   - training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %   - training_data(n,k).handPos(d,t) (d = dimension [1-2], t = time)

    % Initialize variables
    num_neurons = size(training_data(1,1).spikes, 1); % 98 neurons
    num_trials = size(training_data, 1); % Number of trials
    num_angles = size(training_data, 2); % Number of angles
    modelParameters = struct(); % Initialize output struct

    % Initialize containers for the features (inputs) and targets (hand positions)
    features = [];
    targets_x = [];
    targets_y = [];

    % Loop through all angles and trials
    for angle = 1:num_angles
        for trial = 1:num_trials
            data = training_data(trial, angle);
            spikes = data.spikes; % Spike train: 98 x N (neurons x time bins)
            handPos = data.handPos; 
           
            for neuron = 1:num_neurons
                % Calculate the total firing rate for all neurons in the trial
                total_spikes = sum(spikes(neuron,:), 2); 
                firing_rate = sum(total_spikes) / size(spikes(neuron,:), 2); 
            
                % Calculate the cumulative Fano factor 
                spike_counts = sum(spikes(neuron,:), 2); 
                mean_spikes = mean(spike_counts); 
                var_spikes = var(spike_counts);  
                fano_factor = var_spikes / mean_spikes; 

                % Combine firing rate and Fano factor into a single feature vector
                trialFeatures = [firing_rate; fano_factor]; % Features for this trial
                trialHandPos_x = handPos(1,:); 
                trialHandPos_y = handPos(2,:);
            end
        features = [features; trialFeatures];
        targets_x = [targets_x; trialHandPos_x];
        targets_y = [targets_y; trialHandPos_y];
           
        end
    end

    % Store the extracted features and targets in modelParameters
    modelParameters.features = features;
    modelParameters.targets_x = targets_x;
    modelParameters.targets_y = targets_y;

    % Train the regressor for x and y hand positions
    % Using linear regression (you can replace it with another regressor if needed)
    modelParameters.regressor_x = fitlm(features', targets_x'); % Linear regression for x
    modelParameters.regressor_y = fitlm(features', targets_y'); % Linear regression for y

    % Optionally, you could store more regressors or other models here (e.g., neural networks)
end
