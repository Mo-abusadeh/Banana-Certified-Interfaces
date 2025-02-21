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

figure;
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
figure;
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
        t_start = (t-1) * bin_size + 1; % setting  the initial snd end time for each time bin (bin1(1ms-10ms) bin2(11ms-20ms))...
        t_end = min(t * bin_size, max_len); 
        spike_counts(t) = spike_counts(t) + sum(spike_times >= t_start & spike_times < t_end);
    end
end

figure;
bar((1:num_bins) * bin_size, spike_counts, 'k'); % Black bars
xlabel('Time (ms)');
ylabel('Spike Count');
title(sprintf('PSTH (Spike Count) - Neuron %d, Angle %d', neuron_id, angle_id));


%% Q4: Hand Positions for trials across different angles


figure; 

trial_id = 30; % taking the first trial for every angle

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

