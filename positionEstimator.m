function [x, y] = positionEstimator(test_data, modelParameters)
    % Extract trial ID and spikes from the test data
    spikes = test_data.spikes; % Size: 98 x N, where N is current time
    num_neurons = size(spikes, 1); % 98 neurons
    num_timebins = size(spikes, 2); % Time steps (current time, t)

    % Calculate the total firing rate for all neurons in the current trial
    total_spikes = sum(spikes, 2); % Sum of spikes for all neurons (98 x 1)
    firing_rate = sum(total_spikes) / num_timebins; % Total spikes / time bins
    
    % Calculate the cumulative Fano factor across all neurons
    spike_counts = sum(spikes, 2); % Spike count for each neuron (98 x 1)
    mean_spikes = mean(spike_counts); % Mean spike count across all neurons
    var_spikes = var(spike_counts);  % Variance of spike counts across all neurons
    fano_factor = var_spikes / mean_spikes; % Cumulative Fano factor for the trial

    % Combine the firing rate and Fano factor into a feature vector
    features = [firing_rate; fano_factor]; % Features for this trial

    % Predict hand positions (x and y) using the trained regressors
    predicted_x = predict(modelParameters.regressor_x, features');
    predicted_y = predict(modelParameters.regressor_y, features');

    % Return the predicted x and y hand positions
    x = predicted_x;
    y = predicted_y;
end

