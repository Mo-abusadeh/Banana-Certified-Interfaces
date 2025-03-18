function modelParameters = positionEstimatorTraining_kalman(trainingData)
    % Extract number of trials and neurons
    numTrials = length(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);

    NBModel = Naive_Bayes(trainingData);
    
    % Define matrices for Kalman filter
    X = []; % State matrix (position, velocity, hidden intention)
    Z = []; % Observation matrix (spike rates features)
    trainingVelocities = [];
    
    % Collect training data
    binSize = 390;
    for trial = 1:numTrials
        for angle = 1:size(trainingData, 2)
            spikes = trainingData(trial, angle).spikes;
            handPos = trainingData(trial, angle).handPos;
            
            % Compute velocities
            velocities = [];
            for dim = 1:2  
                cumsumPos = cumsum(handPos(dim, :), 2);
                velocity = (cumsumPos(:, binSize+1:end) - cumsumPos(:, 1:end-binSize)) / binSize;
                velocity = [zeros(1, binSize), velocity]; 
                velocities = [velocities; velocity]; 
            end
            trainingVelocities = [trainingVelocities, velocities];
            
            % Compute neural features
            cumsumSpikes = cumsum(spikes, 2);
            firingRates = (cumsumSpikes(:, binSize:end) - cumsumSpikes(:, 1:end-binSize+1)) / binSize;
            firingRates = [zeros(numNeurons, binSize-1), firingRates];

            firingRateChanges = diff(firingRates, 1, 2); 
            firingRateChanges = [zeros(numNeurons, 1), firingRateChanges];

            meanFiringRate = mean(firingRates, 2);

            % Predict angle using Naive Bayes Classifier
            angle_predicted = classifyAngle(spikes, NBModel);

            % Introduce hidden intention states (random initialization)
            hiddenStates = 0.5 * randn(2, size(handPos,2)); % 2 hidden states

            for t = 1:size(handPos,2)-1
                X = [X, [handPos(1,t); handPos(2,t); velocities(:,t); hiddenStates(:,t)]];  
                Z = [Z, [firingRates(:,t); firingRateChanges(:,t); meanFiringRate; angle_predicted]];
            end
        end
    end
    
    % Estimate Kalman parameters
    A = (X(:,2:end) * X(:,1:end-1)') / (X(:,1:end-1) * X(:,1:end-1)');  
    A = A * 0.3;
    
    % Add decay to hidden states in transition matrix
    A(5:6, 5:6) = 0.7 * eye(2); % Hidden states decay over time
    
    W = cov(X(:,2:end)' - (A * X(:,1:end-1))');
    W = W + 1e-3 * eye(size(W));
    
    lambda = 1e-3;
    H = (Z * X') / (X * X' + lambda * eye(size(X,1)));
    
    Q = cov(Z' - (H * X)');
    Q = Q * 0.7;  
 
    % Store parameters
    modelParameters.A = A;
    modelParameters.W = W;
    modelParameters.H = H;
    modelParameters.Q = Q;
    modelParameters.avgVelocity = mean(trainingVelocities, 2);
    modelParameters.classifier = NBModel;
end    

