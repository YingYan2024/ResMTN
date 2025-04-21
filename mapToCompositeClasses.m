% Create function to map original labels to composite labels
function mapped_labels = mapToCompositeClasses(original_labels, composite_class1, composite_class2)
    mapped_labels = zeros(size(original_labels));
    
    % Map to composite class 1
    for i = 1:length(composite_class1)
        mapped_labels(original_labels == composite_class1(i)) = 1;
    end
    
    % Map to composite class 2
    for i = 1:length(composite_class2)
        mapped_labels(original_labels == composite_class2(i)) = 2;
    end
    
    % Any classes not in either composite class will be labeled 0
    % These can be excluded from accuracy calculations if needed
end