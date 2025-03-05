%%% Team Members: WRITE YOUR TEAM MEMBERS' NAMES HERE
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.


function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  

  %%%%%%%%POPULATION VECTOR%%%%%%%%%%%%%
  % Calculate preferred directions for all neurons
preferred_dirs = zeros(num_neurons, 1);

% For each neuron, find its preferred direction
for neuron = 1:num_neurons
    % Initialize arrays to store average firing rates for each angle
    avg_rates = zeros(1, num_angles);
    
    % Calculate average firing rate for each angle
    for angle = 1:num_angles
        total_rate = 0;
        valid_trials = 0;
        
        % Average across all trials for this angle
        for trial = 1:num_trials
            spike_data = training_data(trial, angle).spikes(neuron, :);
            if ~isempty(spike_data)
                % Calculate firing rate (spikes per second)
                rate = sum(spike_data) / (length(spike_data) / 1000);
                total_rate = total_rate + rate;
                valid_trials = valid_trials + 1;
            end
        end
        
        % Calculate average rate if we have valid trials
        if valid_trials > 0
            avg_rates(angle) = total_rate / valid_trials;
        end
    end
    
    % Find the angle with maximum firing rate
    [~, max_angle] = max(avg_rates);
    
    % Convert angle index to direction in radians
    % Assuming angles correspond to these directions (in radians):
    directions_rad = [30, 70, 110, 150, 190, 230, 310, 350] * pi/180;
    preferred_dirs(neuron) = directions_rad(max_angle);
end

    % Store preferred directions in model parameters
    modelParameters.preferred_dirs = preferred_dirs;



      %%%%%%%%Linear Regression%%%%%%%%%%%%%
      for angle = 1:num_angles
        fprintf('Training model for angle %d\n', angle);
        
        % Initialize data containers for this angle
        all_features = [];
        all_x_positions = [];
        all_y_positions = [];
    
        % For each trial at this angle
        for trial = 1:num_trials
            % Get spike data for this trial
            spike_data = training_data(trial, angle).spikes;
            
            % Get hand position data (only X and Y dimensions)
            hand_pos = training_data(trial, angle).handPos(1:2, :);
            
            % Get trial length (in milliseconds)
            trial_length = size(spike_data, 2);
            
            % Calculate number of bins for this trial
            num_bins = floor(trial_length / bin_size);
            
            % Create containers for binned features
            binned_features = zeros(num_neurons, num_bins);
            
            % Process each time bin
            for bin = 1:num_bins
                % Calculate bin boundaries
                bin_start = (bin-1) * bin_size + 1;
                bin_end = min(bin * bin_size, trial_length);
                
                % Calculate firing rate for each neuron in this bin
                spike_counts = sum(spike_data(:, bin_start:bin_end), 2);
                binned_features(:, bin) = spike_counts / (bin_end - bin_start + 1) * 1000; % Convert to spikes/second
            end
            
            % Downsample hand positions to match binned features
            resampled_x = zeros(1, num_bins);
            resampled_y = zeros(1, num_bins);
            
            for bin = 1:num_bins
                bin_end = min(bin * bin_size, trial_length);
                resampled_x(bin) = hand_pos(1, bin_end);
                resampled_y(bin) = hand_pos(2, bin_end);
            end
            
            % Append data from this trial to overall dataset for this angle
            all_features = [all_features, binned_features];
            all_x_positions = [all_x_positions, resampled_x];
            all_y_positions = [all_y_positions, resampled_y];
        end
        
        % Prepare data for regression
        X = all_binned_features'; % Samples × Features matrix
        
        % Add bias term (intercept)
        X = [ones(size(X, 1), 1), X];
        
        % Train linear regression model for X position
        Y_x = all_x_positions';
        beta_x = (X' * X) \ (X' * Y_x); % Ordinary least squares solution
        
        % Train linear regression model for Y position
        Y_y = all_y_positions';
        beta_y = (X' * X) \ (X' * Y_y); % Ordinary least squares solution
        
        % Store coefficients for this angle
        modelParameters.angle(angle).beta_x = beta_x;
        modelParameters.angle(angle).beta_y = beta_y;
        modelParameters.num_neurons = num_neurons;
        modelParameters.bin_size = bin_size;
        modelParameters.angles = 1:num_angles;
        modelParameters.directions_rad = [30, 70, 110, 150, 190, 230, 310, 350] * pi/180;
      end



end

function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  

  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end

