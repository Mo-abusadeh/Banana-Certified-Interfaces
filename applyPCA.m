function [reduced_features, pca_model] = applyPCA(features, variance_threshold)
    % Applies PCA to features, keeping components that explain desired variance
    %
    % Inputs:
    %   features - Feature matrix (samples x features)
    %   variance_threshold - Proportion of variance to retain (e.g., 0.95)
    %
    % Outputs:
    %   reduced_features - Dimensionality-reduced features
    %   pca_model - Structure with PCA parameters for future use
    
    if nargin < 2
        variance_threshold = 0.95;
    end
    
    % Standardize features
    [z, mu, sigma] = zscore(features);
    
    % Apply PCA
    [coeff, score, ~, ~, explained] = pca(z);
    
    % Determine number of components to retain
    cumulative_var = cumsum(explained) / sum(explained);
    num_components = find(cumulative_var >= variance_threshold, 1);
    
    % Select the top components
    reduced_features = score(:, 1:num_components);
    
    % Store PCA model for future use
    pca_model = struct();
    pca_model.mu = mu;
    pca_model.sigma = sigma;
    pca_model.coeff = coeff(:, 1:num_components);
    pca_model.explained = explained(1:num_components);
    
    fprintf('PCA reduced dimensions from %d to %d features (%.1f%% variance retained)\n', ...
        size(features, 2), num_components, sum(explained(1:num_components)));
end