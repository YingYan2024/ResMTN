function [selectedFeatures, selectedLabels] = selectClasses(allFeatures, class1List, class2List, samples1, samples2)
    % allFeatures has shape [featureDimension, sampleCount]
    % class lists contain indices of classes (1-5 for A-E)
    % samples1, samples2 are the number of samples to select from each group
    
    % For class group 1
    group1Features = [];
    for i = 1:length(class1List)
        classIdx = class1List(i);
        startIdx = (classIdx-1)*100 + 1;
        endIdx = classIdx*100;
        group1Features = [group1Features, allFeatures(:, startIdx:endIdx)];
    end
    
    % For class group 2
    group2Features = [];
    for i = 1:length(class2List)
        classIdx = class2List(i);
        startIdx = (classIdx-1)*100 + 1;
        endIdx = classIdx*100;
        group2Features = [group2Features, allFeatures(:, startIdx:endIdx)];
    end
    
    % Random sampling if needed
    totalSamples1 = size(group1Features, 2);
    totalSamples2 = size(group2Features, 2);
    
    if totalSamples1 > samples1
        % Random selection without replacement
        indices = randperm(totalSamples1, samples1);
        group1Features = group1Features(:, indices);
    end
    
    if totalSamples2 > samples2
        % Random selection without replacement
        indices = randperm(totalSamples2, samples2);
        group2Features = group2Features(:, indices);
    end
    
    % Combine features and create labels
    selectedFeatures = [group1Features, group2Features];
    selectedLabels = [ones(1, size(group1Features, 2)), ones(1, size(group2Features, 2))*2];
end