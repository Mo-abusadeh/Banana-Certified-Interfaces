load ('monkeydata_training.mat');
%% Exercise 1
% Familiarize yourself with raster plots and for a single trial, compute and display a population
% raster plot. A population raster would have time (bins) on the x-axis and neural units on the y-axis.

spike_train = trial(6,4).spikes; % Spike train recorded from all neuron units on the 6th trial of the 4th reaching angle
n_units = size(spike_train, 1); % Neural units (each row in the spike_train matrix)
figure;
hold on;

% Loop to plot spike times for specified trial
for neuron = 1:n_units
    spike_times = find(spike_train(neuron, :) == 1); % Find time points where spikes occurred for current neuron
    neuron_idx = neuron * ones(size(spike_times)); % Create array of same size as spike_times (each value is the current neuron's index)
    plot(spike_times, neuron_idx, 'k.');
end

xlabel('Time (ms)');
ylabel('Neural Units');
title('Population Raster Plot for a Single Trial');
set(gca, 'YDir', 'reverse'); 
hold off;

%% Exercise 2
% Compute and display a raster plot for one neural unit over many trials.
% Use different colours for different trials to help with the visualization.

colors = jet(100);
num_trials = length(trial(:, 1)); % Number of trials

figure;
hold on;

for n = 1:num_trials % Number of trials
    spike_times = find(trial(n, 1).spikes(10, :) == 1); % Store only time points where spikes occured for the 10th neuron for all trials, first reaching angle
    plot(spike_times, n * ones(size(spike_times)), '.', 'Color', colors(n,:), 'MarkerSize', 10);
end

xlabel('Time (ms)');
ylabel('Trial Number');
title('Raster Plot for 10th Neural Unit');
hold off;

%% Exercise 3
% Compute peri-stimulus time histograms (PSTHs) for different neural units



%% Exercise 4
% Plot hand positions for different trials

n_trials = 10;
colors = jet(n_trials);

figure;
hold on;

for n = 1:n_trials
    hand_pos = trial(n, :).handPos; % Store hand position for all reaching angles at current trial
    plot(hand_pos(1, :), hand_pos(2, :), 'Color', colors(n,:), 'LineWidth', 1.5); % Plot only x,y positions
end

xlabel('X Position (mm)'); ylabel('Y Position (mm)');
title('Hand Position Across Multiple Trials');
legend(arrayfun(@(x) sprintf('Trial %d', x), 1:n_trials, 'UniformOutput', false)); % Generate dynamic legend labels for each trial
hold off;

%% Exercise 5
num_neurons = 5;
bin_size = 20;    % Bin size in ms
reaching_angles = 1:8;  % 8 movement directions
num_trials = size(trial, 1);  % Total number of trials per direction

% Initialize storage for firing rates
firing_rates = zeros(num_neurons, length(reaching_angles));

% Loop over neurons
for neuron_id = 1:num_neurons
    % Loop over movement directions
    for k = reaching_angles
        spike_counts = 0;
        total_time = 0;
        
        % Loop over trials
        for n = 1:num_trials
            spike_train = trial(n, k).spikes(neuron_id, :);
            spike_counts = spike_counts + sum(spike_train); % Count total spikes
            total_time = total_time + length(spike_train);  % Get total time
        end
        
        % Compute mean firing rate (spikes/sec)
        firing_rates(neuron_id, k) = (spike_counts / total_time) * 1000;
    end
end

figure; hold on;
angles_rad = [30 70 110 150 190 230 310 350] * pi / 180; % Convert angles to radians
colors = lines(num_neurons); % Different colors for neurons

for neuron_id = 1:num_neurons
    errorbar(angles_rad, firing_rates(neuron_id, :), std(firing_rates(neuron_id, :)), ...
        '-o', 'Color', colors(neuron_id,:), 'LineWidth', 1.5);
end

xlabel('Movement Direction (radians)');
ylabel('Mean Firing Rate (spikes/sec)');
title('Tuning Curves for Different Neurons');
legend(arrayfun(@(x) sprintf('Neuron %d', x), 1:num_neurons, 'UniformOutput', false));
hold off;
