function [x, y] = positionEstimator(test_data, modelParameters)
    spikes = test_data.spikes;
    cum_spikes = sum(spikes, 2)'; % Compute cumulative spike counts

    % Predict reaching angle using regression model
    predicted_angle = round(predict(modelParameters.angleRegressor, cum_spikes));

    % Ensure predicted angle is within valid range
    predicted_angle = max(1, min(length(modelParameters.angles), predicted_angle));

    % Use velocity model for predicted angle
    Wx = modelParameters.angles(predicted_angle).Wx;
    Wy = modelParameters.angles(predicted_angle).Wy;

    % Predict velocity
    xdot = cum_spikes * Wx;
    ydot = cum_spikes * Wy;

    % Euler integration for position
    start_pos = test_data.startHandPos;
    duration = size(spikes, 2);
    
    x = start_pos(1) + xdot * duration;
    y = start_pos(2) + ydot * duration;
end

