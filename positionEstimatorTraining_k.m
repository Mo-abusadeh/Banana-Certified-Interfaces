function modelParameters = positionEstimatorTraining_k(trainingData)
    % with positionEstimator_k
    
    % Extract number of trials and neurons
    numTrials = length(trainingData);
    numAngles = size(trainingData, 2);
    numNeurons = size(trainingData(1,1).spikes, 1);

    % Struct to store separate Kalman parameters for each angle
    angle_models = struct();

    binSize = 395;
    idealDirections = [cosd(0:45:315); sind(0:45:315)];

    for angle = 1:numAngles
        X = []; % State matrix (position, velocity, acceleration)
        Z = []; % Observation matrix (spike rates)
        trainingVelocities = [];

        for trial = 1:numTrials
            spikes = trainingData(trial, angle).spikes;
            handPos = trainingData(trial, angle).handPos;
            
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

            firingRateChanges = diff(firingRates, 1, 2);
            firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];

            meanVelocity = mean(velocities, 2);
            meanAcceleration = mean(acceleration,2);
            meanFiringRate = mean(firingRates, 2);
            meanFiringRateChange = mean(firingRateChanges, 2);

            % Compute cosine similarity with ideal movement directions
            currentDirection = velocities ./ (vecnorm(velocities) + 1e-6);
            cosTheta = idealDirections' * currentDirection;
            cosTheta = mean(cosTheta, 2);
            
            % Ensure we only collect up to `numTimeSteps`
            numTimeSteps = size(firingRates, 2);  % Should be 617
            X = zeros(18, numTimeSteps);  % Preallocate instead of appending
            Z = zeros(98, numTimeSteps);  % Preallocate instead of appending

            % Loop through time steps correctly
            for t = 1:numTimeSteps
                X(:,t) = [handPos(1,t); handPos(2,t); velocities(:,t); acceleration(:,t); meanVelocity; meanAcceleration; cosTheta];

                %Z(:,t) = [firingRates(:,t); firingRateChanges(:,t)];  % Keep only neuron-dependent features
                %Z(:, t) = mean([firingRates(:, t), firingRateChanges(:, t)], 2);  % Combine features
                Z = [firingRates, firingRateChanges];  % Expand features in columns instead of stacking vertically

            end

            %for t = 1:size(handPos,2)-1
            %    X = [X, [handPos(1,t); handPos(2,t); velocities(:,t); acceleration(:,t); meanVelocity; meanAcceleration; cosTheta]];
            %    Z = [firingRates, ...
            %        firingRateChanges, ...
            %        repmat(meanFiringRate, 1, size(firingRates, 2)), ...
            %        repmat(meanFiringRateChange, 1, size(firingRates, 2))];
            %end
        end  


        % Estimate Kalman parameters
        A = (X(:,2:end) * X(:,1:end-1)') / (X(:,1:end-1) * X(:,1:end-1)');

        W = cov(X(:,2:end)' - (A * X(:,1:end-1))');
        W = W + 1e-3 * eye(size(W));

        lambda = 1e-3;
        H = (Z * X') / (X * X' + lambda * eye(size(X,1)));

        Q = cov(Z' - (H * X)');
        Q = Q * 0.7;


        % **Store Kalman parameters in struct arrays**
        angle_models.A(:,:,angle) = A;  % Store each matrix at the third dimension
        angle_models.W(:,:,angle) = W;
        angle_models.H(:,:,angle) = H;
        angle_models.Q(:,:,angle) = Q;
    end
 
    % Store all angle models
    modelParameters.angle_models = angle_models;
end

