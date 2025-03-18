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

% Select top neurons (top 50%)
num_top_neurons = ceil(length(ranked_neurons) * 0.5); % HYPERPARAMETER
top_neurons = ranked_neurons(1:num_top_neurons);