function [fuzzy_approx_entropy] = fuzzy_approx_entropy(data)
% Calculate fuzzy approximate entropy for a given sample matrix
% Input:
%   data: a 4097 x 200 matrix representing the sample dataset, where each
%         column is a time series of length 4097
% Output:
%   fuzzy_approx_entropy: a 1 x 200 vector representing the fuzzy approximate entropy of each sample

% number of time points and samples
[n, m] = size(data);

% fuzzification parameter
r = 0.2*std(data(:));

% calculate distance matrix
dist = zeros(m, m);
for i = 1:m
    for j = i+1:m
        tmp = data(:, i) - data(:, j);
        dist(i, j) = max(abs(tmp));
        dist(j, i) = dist(i, j);
    end
end

% calculate fuzzy approximate entropy for each sample
fuzzy_approx_entropy = zeros(1, m);
for i = 1:m
    % calculate membership function for all other samples
    membership = exp(-(dist(i, :) - r).^2/(2*r^2));
    % calculate fuzzy approximate entropy
    fuzzy_approx_entropy(i) = -mean(log2(membership(membership > 0)));
end
end
