function [x, y] = positionEstimator_v2(test_data, modelParameters)

    % Extract trial ID and spikes from the test data
    spikes = test_data.spikes; % Size: 98 x N, where N is current time
    num_neurons = size(spikes, 1); % 98 neurons
    num_timebins = size(spikes, 2); % Time steps (current time, t)

    for trial = 1:num_trials
        spikes = spikes(trial); % spikes trains for 98 neurons

        % Average over timesteps
        duration = size(spikes(1,:),2);
        nb_bins = 10;
        timesteps = floor(duration/nb_bins); % time bins over which we average the features
            
        for n = 1:nb_bins
            % Compute start and end times
            t1 = (n-1) * timesteps;
            t1 = t1 + 1;
            t2 = t1 + (timesteps-1);

            % Calculate the total firing rate for all neurons in the trial
            total_spikes = sum(spikes(:,t1:t2), "all");
            firing_rate = total_spikes / size(spikes(:,t1:t2)', 2);
            
            size(spikes(:,t1:t2));

            % Calculate the Fano factor  
            mean_spikes = mean(spikes(:,t1:t2),"all"); 
            var_spikes = var(spikes(:,t1:t2),0,"all");
            %var_spikes = mean(var_spikes);
            % fano_factor = var_spikes / mean_spikes
            
            binned_features(n,:) = [firing_rate, total_spikes, mean_spikes, var_spikes];
                                % Array of features (over all neurons per time bins) ~10x4
        end
        trial_features(trial,:,:) = [binned_features];
    end 
    
end

