function [x, y] = positionEstimator_k(testData, modelParameters)
    % Use in combo with positionEstimatorTraining_k
    % been trying to debug the size of Z / the downstream effects it has

    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 395;
    numTrials = size(spikes, 3);

    selected_angles = zeros(1, numTrials);
    x_positions = zeros(1, numTrials);
    y_positions = zeros(1, numTrials);

    for trial = 1:numTrials
        trial_spikes = spikes(:, :, trial);

        cumsumSpikes = cumsum(trial_spikes, 2);
        firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
        firingRates = [zeros(numNeurons, binSize-1), firingRates];

        Z = firingRates;

        best_rmse = Inf;
        best_angle = 1;

        for angle = 1:8
            % **Retrieve Kalman Parameters Correctly**
            A = modelParameters.angle_models.A(:,:,angle);
            W = modelParameters.angle_models.W(:,:,angle);
            H = modelParameters.angle_models.H(:,:,angle);
            Q = modelParameters.angle_models.Q(:,:,angle);
        
            % Apply Kalman filter
            x_est = [testData.startHandPos(1, trial); testData.startHandPos(2, trial); zeros(16,1)];
            P_est = eye(18);

            X_pred = A * x_est;
            P_pred = A * P_est * A' + W;
            lambda = 1e-6;  
            S = H * P_pred * H' + Q + lambda * eye(size(Q));
            K = (P_pred * H') / S;

            X_est = X_pred + K * (Z - H * X_pred);
            P_est = P_pred - K * H * P_pred;

            % Compute RMSE
            x_predicted = X_est(1, end);
            y_predicted = X_est(2, end);
            rmse = sqrt(x_predicted^2 + y_predicted^2);

            if rmse < best_rmse
                best_rmse = rmse;
                best_angle = angle;
            end
        end

        % Use best angle model
        selected_angles(trial) = best_angle;
        x_positions(trial) = X_est(1, end);
        y_positions(trial) = X_est(2, end);
    end

    x = x_positions(end);
    y = y_positions(end);
end

