function data_expan = Expan(data, power)
    [p, n] = size(data);
    
    % Initialize with original features
    data_expan = data;
    
    % For each power level from 2 to the specified power
    for curr_power = 2:power
        % Get all combinations with repetition for current power
        combinations = generate_combinations(p, curr_power);
        
        % For each combination, compute the product for all samples
        num_combinations = size(combinations, 1);
        new_features = zeros(num_combinations, n);
        
        for i = 1:num_combinations
            % Get current combination
            indices = combinations(i, :);
            
            % Compute product for all samples
            prod_features = ones(1, n);
            for j = 1:curr_power
                prod_features = prod_features .* data(indices(j), :);
            end
            
            % Store the result
            new_features(i, :) = prod_features;
        end
        
        % Add new features to expanded data
        data_expan = [data_expan; new_features];
    end
    data_expan = [ones(1, n);data_expan];
end

function C = generate_combinations(n, k)
    % Generate combinations with repetition allowed
    % n: number of elements to choose from (1 to n)
    % k: number of elements to select
    
    % Initialize first combination
    c = ones(1, k);
    
    % Pre-allocate maximum number of combinations
    max_combs = nchoosek(n+k-1, k);
    C = zeros(max_combs, k);
    C(1, :) = c;
    count = 1;
    
    % Generate all combinations
    while true
        % Find rightmost position that can be incremented
        i = k;
        while i >= 1 && c(i) == n
            i = i - 1;
        end
        
        % If no position can be incremented, we're done
        if i == 0
            break;
        end
        
        % Increment position i
        c(i) = c(i) + 1;
        
        % Reset all positions to the right
        c(i+1:k) = c(i);
        
        % Store combination
        count = count + 1;
        C(count, :) = c;
    end
    
    % Return only valid combinations
    C = C(1:count, :);
end