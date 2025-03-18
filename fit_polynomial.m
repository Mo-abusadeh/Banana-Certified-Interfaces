function p_coeffs = fit_polynomial(x, y, n)
    % n : maximum order
    % p : coefficient vector

    % Construct the Vandermonde matrix
    V = fliplr(vander(x));  % Flip so highest degree is first
    V = V(:, 1:n+1);  % Only take required columns for degree n

    %s1 = size(V' * V)
    %s2 = size(V' * y')
    
    
    % Solve the normal equations using least squares
    p_coeffs = (V' * V) \ (V' * y');  % Solve for coefficients
end

% function [pred] = predict(test_data, modelParameters, features)
% 
%     for angle = 1:8
% 
%     end
% end
