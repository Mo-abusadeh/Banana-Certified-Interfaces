function [modelParameters] = positionEstimatorTraining_v2(training_data)
    % Arguments:
    % - training_data: A 2D matrix of struct data with trials and angles
    %   - training_data(n,k).trialId      unique trial ID
    %   - training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %   - training_data(n,k).handPos(d,t) (d = dimension [1-2], t = time)

    % Initialize variables
    num_neurons = size(training_data(1,1).spikes, 1); % 98 neurons
    num_trials = size(training_data, 1); % Number of trials (100)
    num_angles = size(training_data, 2); % Number of angles (8)
    handPos = struct();
    modelParameters = struct();   

    features = [];
    targets_x = [];
    %targets_y = [];

    for angle = 1:num_angles
        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes; % spikes trains for 98 neurons
            trial_handPos = training_data(trial,angle).handPos(1:2, :); % handPos (x,y) 2xN (N time instances)
            for neuron = 1:num_neurons 
                % Calculate the total firing rate for all neurons in the trial
                total_spikes = sum(spikes(neuron,:), 2); 
                firing_rate = sum(total_spikes) / size(spikes(neuron,:), 2); 
            
                % Calculate the cumulative Fano factor  
                mean_spikes = mean(spikes(neuron,:)); 
                var_spikes = var(spikes(neuron,:));  
                fano_factor = var_spikes / mean_spikes;
                
                neuron_features(neuron, :) = [firing_rate, total_spikes, mean_spikes, var_spikes, fano_factor]; % Array of features (neurons specific) 98x4
            end
            trial_features(trial,:,:) = [neuron_features]; % population specific features for every trial 
        end
        features(angle,:,:,:) = [trial_features];
        handPos(angle, trial).positions_x = trial_handPos(1,:)';
        handPos(angle, trial).positions_y = trial_handPos(2,:)';
    end


    % Train the regressor 
    for angle = 1:num_angles
        for trial = 1:num_trials
            duration = size(handPos(angle, trial).positions_x,1);
            for t = 2:duration
                curr_x = handPos(angle, trial).positions_x(t);
                curr_y = handPos(angle, trial).positions_y(t);
                %size(curr_x)
                prev_x = handPos(angle, trial).positions_x(t-1);
                prev_y = handPos(angle, trial).positions_y(t-1);
                %size(prev_x)
                slope = (curr_y - prev_y) / (curr_x - prev_x)
                

            end
        end
    end

end
