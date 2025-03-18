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

            % Average over timesteps
            duration = size(trial_handPos(1,:),2);
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
                binned_handPos_x(n,:) = mean(trial_handPos(1,t1:t2));
                binned_handPos_y(n,:) = mean(trial_handPos(2,t1:t2));
            end
            trial_features(trial,:,:) = [binned_features];
            trial_handPos_x(trial, :, :) = [binned_handPos_x];
            trial_handPos_y(trial, :, :) = [binned_handPos_y];
        end
        features(angle,:,:,:) = [trial_features];
        handPos_x(angle,:,:,:) = [trial_handPos_x];
        handPos_y(angle,:,:,:) = [trial_handPos_y];
    end

    % Train the regressor
    
    for angle = 1:num_angles
        total_coeffs_x = zeros(4,1);
        total_coeffs_y = zeros(4,1);
        for trial = 1:num_trials
            % Extract training data
            training_features(1,:) = features(angle, trial, :, 1); % firing rate
            training_features(2,:) = features(angle, trial, :, 2); % total spikes

            training_handPos_x(1,:) = handPos_x(angle, trial,:);
            training_handPos_y(1,:) = handPos_y(angle, trial,:);

            % Find polynomial coefficients for each feature
            coeffs_FR_x = fit_polynomial(training_features(1,:), training_handPos_x, 3);
            coeffs_FR_y = fit_polynomial(training_features(1,:), training_handPos_y, 3);
            
            % Running mean over trials (adjusts at each iteration)
            total_coeffs_x = total_coeffs_x + coeffs_FR_x/num_trials;
            total_coeffs_y = total_coeffs_y + coeffs_FR_y/num_trials;
        end 
        %modelParameters(angle).totalcoeffs_x = total_coeffs_x;
        %modelParameters(angle).totalcoeffs_y = total_coeffs_y;
        modelParameters(angle).features = features;
        modelParameters(angle).handPos_x = handPos_x;
        modelParameters(angle).handPos_y = handPos_y;
    end
end
