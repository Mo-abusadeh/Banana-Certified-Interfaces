function modelParameters = positionEstimatorTraining_d(trainingData)
    % calculates 8 model parameters (Kalman) for each of the 8 angles
    
    % Extract number of trials, angles, and neurons
    numTrials = length(trainingData);
    numAngles = size(trainingData, 2);
    numNeurons = size(trainingData(1,1).spikes, 1);

    % Struct to store separate Kalman parameters for each angle
    angle_models = struct();

    binSize = 390;
    idealDirections = [cosd(0:45:315); sind(0:45:315)];

    for angle = 1:numAngles
        X = []; % State matrix (position, velocity, acceleration)
        Z = []; % Observation matrix (spike rates)
        trainingVelocities = [];

        for trial = 1:numTrials
             spikes = trainingData(trial, angle).spikes;
            handPos = trainingData(trial, angle).handPos;
            
            % % Compute velocities
            % velocities = diff(handPos(1:2, :)')';
            % velocities = [velocities, velocities(:, end)]; % Repeat last velocity to match dimensions
            % trainingVelocities = [trainingVelocities, velocities];

            velocities = [];
            for dim = 1:2  
                cumsumPos = cumsum(handPos(dim, :), 2);
                velocity = (cumsumPos(:, binSize+1:end) - cumsumPos(:, 1:end-binSize)) / binSize;
                velocity = [zeros(1, binSize), velocity]; 
                velocities = [velocities; velocity]; 
            end
            trainingVelocities = [trainingVelocities, velocities];
            acceleration = diff(velocities, 1, 2);
            acceleration = [zeros(2, 1), acceleration];
            
            % Compute firing rates
            cumsumSpikes = cumsum(spikes, 2);
            firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
            firingRates = [zeros(numNeurons, binSize-1), firingRates];

            firingRateChanges = diff(firingRates, 1, 2); % First derivative of firing rates
            firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];

            % Compute Fano Factor
            spikeCountVariance = var(spikes, 0, 2);
            spikeCountMean = mean(spikes, 2);
            fanoFactor = spikeCountVariance ./ (spikeCountMean + 1e-6);
            fanoFactor = repmat(fanoFactor, 1, size(firingRates,2));
            %size_f = size(fanoFactor)
                
            meanVelocity = mean(velocities, 2); 
            meanFiringRate = mean(firingRates, 2);

            meanAcceleration = mean(acceleration,2);
            meanFiringRateChange = mean(firingRateChanges, 2);

            currentDirection = velocities ./ (vecnorm(velocities) + 1e-6);
            cosTheta = idealDirections' * currentDirection; % (8xT)
            cosTheta = mean(cosTheta, 2);
            
            for t = 1:size(handPos,2)-1
                X = [X, [handPos(1,t); handPos(2,t); velocities(:,t); acceleration(:,t); meanVelocity; meanAcceleration; cosTheta]]; 
                Z = [Z, [firingRates(:,t); firingRateChanges(:,t); meanFiringRate; meanFiringRateChange]];

            end
        end
    
        % Estimate Kalman parameters
        A = (X(:,2:end) * X(:,1:end-1)') / (X(:,1:end-1) * X(:,1:end-1)');
        A = A * 0.3;
        W = cov(X(:,2:end)' - (A * X(:,1:end-1))');
        W = W + 1e-3 * eye(size(W));
        %H = (Z * X') / (X * X');
        %H = (Z * X') / (X * X' + 0.01 * eye(size(X,1)));
        lambda = 1e-3; % Regularization parameter
        H = (Z * X') / (X * X' + lambda * eye(size(X,1)));
        Q = cov(Z' - (H * X)');
        Q = Q * 0.7; 

        % Store models per angle
        angle_models(angle).A = A;
        angle_models(angle).W = W;
        angle_models(angle).H = H;
        angle_models(angle).Q = Q;
    end

    % Store all angle-specific models
    modelParameters.angle_models = angle_models;
end

