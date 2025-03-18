function [x, y, modelParameters] = positionEstimator_t(testData, modelParameters)
    % Use in combo with positionEstimatorTraianing_d
    % third attempt

    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 395;

    % Compute smoothed firing rates
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];

    firingRateChanges = diff(firingRates, 1, 2); % First derivative of firing rates
    firingRateChanges = [zeros(numNeurons, 1), firingRateChanges]; % Pad with zeros
    meanFiringRate = mean(firingRates, 2);
    meanFiringRateChange = mean(firingRateChanges, 2);

    % Construct observation matrix Z
    Z = [firingRates; firingRateChanges; 
         repmat(meanFiringRate, 1, size(firingRates, 2)); 
         repmat(meanFiringRateChange, 1, size(firingRateChanges, 2))];

    % **Detect if This Is a New Trial Using `startHandPos`**
    persistent lastHandPos;
    if isempty(lastHandPos) || norm(testData.startHandPos - lastHandPos) > 1e-3
        disp('🔄 New trial detected (Hand position reset)');
        best_rmse = Inf;
        best_angle = 1;
        lastHandPos = testData.startHandPos;  % Update stored position
    end

    % **Iterate Through Angles to Select Best Model Per Trajectory**
    best_rmse = Inf;
    best_angle = 1;
    best_A = [];
    best_W = [];
    best_H = [];
    best_Q = [];

    for angle = 1:8
        % Retrieve Kalman parameters for this angle
        A = modelParameters.angle_models(angle).A;
        W = modelParameters.angle_models(angle).W;
        H = modelParameters.angle_models(angle).H;
        Q = modelParameters.angle_models(angle).Q;

        % Apply Kalman filter
        x_est = [testData.startHandPos(1); testData.startHandPos(2); zeros(16,1)];
        P_est = eye(18);

        X_pred = A * x_est;
        P_pred = A * P_est * A' + W;
        lambda = 1e-6;

        % **Regularization check to prevent singularity issues**
        if rcond(H * P_pred * H' + Q) < 1e-12
            lambda = 1e-3;
        end

        S = H * P_pred * H' + Q + lambda * eye(size(Q));
        K = (P_pred * H') / S;
        X_est = X_pred + K * (Z - H * X_pred);
        P_est = P_pred - K * H * P_pred;

        % **Compute RMSE for This Angle**
        x_predicted = X_est(1, end);
        y_predicted = X_est(2, end);
        movement_mag = norm([testData.startHandPos(1), testData.startHandPos(2)]);
        rmse = sqrt(((x_predicted - testData.startHandPos(1))^2 + (y_predicted - testData.startHandPos(2))^2) / movement_mag);

        % **Debug: Print RMSE per angle**
        disp(['Angle ', num2str(angle), ' RMSE: ', num2str(rmse)]);

        % **Reset RMSE Comparison on New Trial**
        if isempty(lastHandPos) || norm(testData.startHandPos - lastHandPos) > 1e-3
            best_rmse = Inf;
            best_angle = 1;
        end

        % **Store Best Angle for This Trajectory**
        if rmse < best_rmse
            best_rmse = rmse;
            best_angle = angle;
            best_A = A;
            best_W = W;
            best_H = H;
            best_Q = Q;
        end
    end

    % **Recompute Final Position Using Best Angle Model**
    x_est = [testData.startHandPos(1); testData.startHandPos(2); zeros(16,1)];
    P_est = eye(18);

    X_pred = best_A * x_est;
    P_pred = best_A * P_est * best_A' + best_W;
    lambda = 1e-6;
    S = best_H * P_pred * best_H' + best_Q + lambda * eye(size(best_Q));
    K = (P_pred * best_H') / S;
    X_est = X_pred + K * (Z - best_H * X_pred);
    P_est = P_pred - K * best_H * P_pred;

    % **Final estimated position**
    x = X_est(1, end);
    y = X_est(2, end);

    % **Debug: Confirm selected model per trajectory**
    disp(['🚀 Detected New Trial? ', num2str(norm(testData.startHandPos - lastHandPos) > 1e-3), ...
          ', Selected Angle: ', num2str(best_angle)]);
end
