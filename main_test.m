%% Understanding the data

%/
% 100 trials
% 8 reaching angles
% 98 neural units
% 182 reaches per angle
% /%

% Raster Plot for a single reach angle
%trial(n,k).spikes(i,:) % i:unit | n:trial | k:reaching angle

% Arrays' Properties (length)
% N_trials = length(trial(:,reaching_angle)); % number of trials
% N_reachingAngles = 
% reaching_angle = 1;
% 
% T_spiketimes = trial(:,reaching_angle).spikes(N_trials,:);
% L_sampleLength = length(trial(:,reaching_angle).spikes(1,:));
% 
% rasterplot(T_spiketimes, N_trials, L_sampleLength);




%%
% Load the data
clc; clear all; close all;
load('monkeydata_training.mat')

%%
% Reformat data into multidimensional array
num_neurons = size(trial(1,1).spikes, 1); 
num_trials = size(trial, 1);
num_angles = size(trial, 2);

% Find the maximum trial length
max_len = 0;
for angles = 1:num_angles
    for trials = 1:num_trials
        max_len = max(max_len, size(trial(trials,angles).spikes, 2)); 
    end
end

spikeMatrix = nan(num_neurons, max_len, num_trials, num_angles);

% Convert struct into multidimensional matrix (easier to manipulate)
for angles = 1:num_angles
    for trials = 1:num_trials
        trial_length = size(trial(trials,angles).spikes, 2); % recording length (in samples)
        spikeMatrix(:,1:trial_length,trials,angles) = trial(trials,angles).spikes; 
    end
end

neuron_id = 1;
angle_id = 1; 

%% Q1: Raster Plot for a single Trial
% Raster plot for all populations over one trial

trial_id = 100;

spike_data = spikeMatrix(:,:,trial_id, angle_id); % 98 x Time matrix (not all trials have 672 samples)
trial_length = size(spike_data, 2);

figure('Name', 'Q1: Raster Plot for a Single Trial');
hold on;
for neuron_id = 1:size(spike_data, 1) 
    spike_times = find(spike_data(neuron_id, :) == 1); 
    y_neuron = neuron_id * ones(size(spike_times));
    scatter(spike_times, y_neuron, 5, 'k', 'filled');
end

xlabel('Time (ms)');
ylabel('Neuron ID');
title(sprintf('Population Raster Plot - Trial %d, Angle %d', trial_id, angle_id));
ylim([0, size(spike_data, 1) + 1]); % Ensure all neurons are visible
xlim([0, trial_length]); % Adjust x-axis based on trial length
hold off;


%% Q2
% Raster plot for one neuron across all trials
figure('Name', 'Q2: Raster plot for one neuron across all trials');
hold on;

colors = jet(num_trials); 

for n = 1:num_trials
    spike_times = find(trial(n,angle_id).spikes(neuron_id, :) == 1); 
    y_trial = n * ones(size(spike_times)); 
    scatter(spike_times, y_trial, 10, colors(n, :), 'filled');
end

xlabel('Time (ms)');
ylabel('Trial Number');
title(sprintf('Raster Plot - Neuron %d, Angle %d', neuron_id, angle_id));
ylim([0 num_trials+1]);
xlim([0 max_len]); 
colorbar;
hold off;

%% Q3
% PSTH plot for a given neuron and angle

% Define bin size
bin_size = 10; % 10 ms bins
num_bins = floor(max_len / bin_size); % Number of bins

% counter
spike_counts = zeros(1, num_bins);

% Loop through trials and bin spikes
for n = 1:num_trials
    spike_times = find(trial(n, angle_id).spikes(neuron_id, :) == 1); 
    for t = 1:num_bins
        t_start = (t-1) * bin_size + 1; % setting  the initial and end time for each time bin (bin1(1ms-10ms) bin2(11ms-20ms))...
        t_end = min(t * bin_size, max_len); 
        spike_counts(t) = spike_counts(t) + sum(spike_times >= t_start & spike_times < t_end);
    end
end

figure('Name', 'Q3: PSTH plot for a given neuron and angle');
bar((1:num_bins) * bin_size, spike_counts, 'k'); % Black bars
xlabel('Time (ms)');
ylabel('Spike Count');
title(sprintf('PSTH (Spike Count) - Neuron %d, Angle %d', neuron_id, angle_id));


%% Q4: Hand Positions for trials across different angles

figure('Name', 'Q4: Hand Positions for trials across different angles'); 

trial_id = 1; % taking the first trial for every angle

colour_pos = jet(num_angles);

for angle = 1:num_angles
    
    % Get hand position data for the chosen trial across all angles
    handPos = trial(trial_id, angle).handPos;
    
    % Get x and y coordinates
    x_coord = handPos(1,:);
    y_coord = handPos(2,:);

    % Plot hand positions
    scatter(x_coord, y_coord, 'Color',colour_pos(1,:))
    hold on;

end 

title(sprintf('Trial %d', trial_id));
xlabel('X Position (mm)');
ylabel('Y Position (mm)');

%% Q5
% For several neurons, plot tuning curves for movement direction

figure('Name', 'Q5: Tuning Curves for Movement Direction');
hold on;

angles = linspace(0, 360, num_angles+1); % create x axis of plot with angles in degrees
angles = angles(1:end-1); 

num_neurons_to_plot = 5; % number of neurons we want to plot
selected_neurons = randperm(num_neurons, num_neurons_to_plot); % select random neurons

colours = jet(num_neurons_to_plot);

for i = 1:num_neurons_to_plot
    neuron_id = selected_neurons(i);
    mean_firing_rates = zeros(1, num_angles); % store mean firing rate per angle
    std_firing_rates = zeros(1,num_angles); % store standard deviations per angle

    firing_rates = zeros(num_angles, num_trials);

    for angle_id = 1:num_angles
        total_spikes = 0;
        for trial_id = 1:num_trials
            % compute total spikes for a neuron across all trials for certain angle
            total_spikes = total_spikes + sum(trial(trial_id, angle_id).spikes(neuron_id, :)); 
            
            % store firing rates 
            firing_rates(angle_id, trial_id) = sum(trial(trial_id, angle_id).spikes(neuron_id, :)) /...
                size(trial(trial_id, angle_id).spikes(neuron_id, :),2);
        end
        mean_firing_rates(angle_id) = total_spikes / num_trials; % compute average firing rate per angle
        std_firing_rates(angle_id) = std(firing_rates(angle_id, :)); % compute variability across trials 
    end

    plot(angles, mean_firing_rates, '-o', 'LineWidth', 2, 'Color', colours(i,:), ...
        'DisplayName', sprintf('Neuron %d', neuron_id));
    errorbar(angles, mean_firing_rates, std_firing_rates, 'o', ...
        'Color', colours(i, :), 'CapSize', 10, 'LineWidth', 1.5, ...
        'HandleVisibility', 'off'); % plot error bars and hide from legend
end

xlabel('Movement Direction (°)');
ylabel('Average Firing Rate (spikes/sec)');
title('Tuning Curves for Multiple Neurons');
legend();
grid on;
hold off;


%% Q7: Population Vector Algorithm Implementation

% should turn into a separate function later

% Get number of intervals (timestamps)
num_timestamps = size(trial(1,1).spikes,2);

%initialise preferred directions and firing rates
preferred_dirs = zeros(num_neurons,1);
firing_rates = zeros(num_neurons, num_timestamps);

trial_id = 1;

%% Q7 Part 1: Determine Preferred Directions
for neuron = 1:num_neurons

    % initialising firing rate sums and counts for every directional bin
    bin_sums = zeros(num_bins,1);
    bin_counts = zeros(num_bins,1);

    % Looping through trials for every angle
    for angle = 1:num_angles
        for trial_id = 1:num_trials

            % Get Spikes for every trial
            spikes = trial(trial_id, angle).spikes(neuron, :);

            % Average firing rate for the trial
            avg_firing_rate = sum(spikes) / (num_timestamps * 1000); % converted to spikes per second

            % Add to direction bin
            bin_sums(angle) = bin_sums(angle) + avg_firing_rate;
            bin_counts(angle) = bin_counts(angle) + 1;
        end
    end

    % Average firing rate for every direction bin
    bin_avgs = bin_sums ./ bin_counts;

    % bin with highest avg firing rate
    [~, preferred_bin] = max(bin_avgs);

    % convert preferred bin into an angle (radians)
    preferred_dirs(neuron) = (preferred_bin - 1) * 2 * pi / num_bins;
end 


%% Q7 Part 2: Calculate Firing Rates

window_size = 100; % ±50 ms

for neuron = 1:num_neurons
    for timestamp = 1:num_timestamps

        % Find start and end of windows
        window_start = max(1,timestamp - window_size/2);
        window_end = min(num_timesteps);


        %Count the spikes in the window 
        spike_count = sum(spikeMatrix(neuron, window_start:window_end, trial_id,angle));

        % Calculate firing rate firing rate (spikes per second)
        firing_rates(neuron, timestamp) = spike_count / (window_size * 1000);
    end 
end 


%% Q7 Part 3 Build a Population Vector
population_vectors = zeros(2, num_timesteps);

for timestep = 1:num_timesteps
    % Initialize the population vector for this timestep
    population_vector = [0; 0];

    for neuron = 1:num_neurons
        % Get the preferred direction and firing rate for this neuron
        preferred_dir = preferred_dirs(neuron);
        firing_rate = firing_rates(neuron, timestep);

        % Create a vector in the preferred direction with length proportional to the firing rate
        neuron_vector = [cos(preferred_dir); sin(preferred_dir)] * firing_rate;

        % Add to the population vector
        population_vector = population_vector + neuron_vector;
    end

    % Store the population vector for this timestep
    population_vectors(:, timestep) = population_vector;
end


%% Build a regression model to predict the trajectories
