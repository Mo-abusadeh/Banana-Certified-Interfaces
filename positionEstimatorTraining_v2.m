function modelParameters = positionEstimatorTraining_v3(training_data)
    num_trials = size(training_data, 1);
    num_angles = size(training_data, 2);
    
    
    modelParameters = struct();

    % Store data for angle prediction
    X_angle = [];
    y_angle = [];

    for angle = 1:num_angles
        X_data = [];
        xdot_data = [];
        ydot_data = [];
        
        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes;
            cum_spikes = sum(spikes, 2)';

            % Store data for regression-based angle prediction / SVM 
            X_angle = [X_angle; cum_spikes];
            y_angle = [y_angle; angle]; % Encode the reaching angle as output

            % Compute velocity
            start_pos = training_data(trial, angle).handPos(1:2, 1);
            final_pos = training_data(trial, angle).handPos(1:2, end);
            duration = size(training_data(trial, angle).handPos, 2);
            v = (final_pos - start_pos) / duration;

            X_data = [X_data; cum_spikes];
            xdot_data = [xdot_data; v(1)];
            ydot_data = [ydot_data; v(2)];
        end

        % Train velocity models per angle
        modelParameters.angles(angle).Wx = pinv(X_data) * xdot_data;
        modelParameters.angles(angle).Wy = pinv(X_data) * ydot_data;
    end

    %modelParameters.angleRegressor = fitlm(X_angle, y_angle, 'linear'); % Linear regression
    modelParameters.angleRegressor = fitcecoc(X_angle, y_angle); % Multiclass SVM

end

