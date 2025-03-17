function [x, y] = positionEstimator_d(testData, modelParameters)
    % Also use in combo with posiionEstimatorTraining_d
    % second attempt at 8 model parameters (per angle)

    spikes = testData.spikes;
    numNeurons = size(spikes, 1);
    binSize = 395;

    % Compute smoothed firing rates
    cumsumSpikes = cumsum(spikes, 2);
    firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
    firingRates = [zeros(numNeurons, binSize-1), firingRates];
    %Z = firingRates(:, 1:T);
    firingRateChanges = diff(firingRates, 1, 2); % First derivative of firing rates
    firingRateChanges = [zeros(numNeurons, 1), firingRateChanges]; % Pad with zeros
    meanFiringRate = mean(firingRates, 2);
    % % Compute variance and mean of firing rates over time
    % spikeCountVariance = var(spikes, 0, 2);
    % spikeCountMean = mean(spikes, 2);
    % fanoFactor = spikeCountVariance ./ (spikeCountMean + 1e-6); 
    meanFiringRateChange = mean(firingRateChanges, 2);


    Z = [firingRates; firingRateChanges; repmat(meanFiringRate, 1, size(firingRates, 2)); repmat(meanFiringRateChange, 1, size(firingRateChanges, 2))];  
   

    best_rmse = Inf;
    best_angle = 1;

    for angle = 1:8
        % Retrieve Kalman parameters for each angle
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

    % Use the best angle model
    x = X_est(1, end);
    y = X_est(2, end);
    disp(['Selected model parameters for angle: ', num2str(best_angle)]);

end

