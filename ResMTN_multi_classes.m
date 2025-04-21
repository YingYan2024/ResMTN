clc;clear;

load Bonn.mat

% Training 5 class, but only test A~D with E.

% Define original labels (1-5 for classes A-E)
original_Labels = [ones(1,100), ones(1,100)*2, ones(1,100)*3, ones(1,100)*4, ones(1,100)*5];

% Define composite classes for testing
composite_class1 = [1, 2, 3, 4]; % Classes A~D
composite_class2 = [5]; % Classes E

[feature,sorted_indices]=feature_selection(feature,original_Labels',24,2);

t0=tic; % Timing the entire process

% Set learning rate and training epochs
testNum=10; % Number of repeated experiments
numEpochs = 1000; % Maximum iterations for each fold in cross-validation
lr = 0.001; % Initial learning rate
k=10;  % 10-fold cross-validation
Acti_type = 4; % Activation function type: 1=RELU, 2=LeakyRELU, 3=Logistic, 4=tanh
Power=2; % Set the highest expansion order for MTN
class_num=5; % Number of original classes (A through E)
composite_class_num=2; % Number of composite classes for testing
shift_num=0.2; % Shift parameter for normalization
total_num = size(feature, 2); % Get actual sample count from feature size
train_num=round(total_num*(k-1)/k); % Size of training set (nine folds)

% Set network structure
feature_size=size(feature,1); % Dimension of feature vector

% Calculate expanded feature size based on polynomial order
expan_size=1;
for i=1:Power
    expan_size=expan_size + prod(feature_size:(feature_size+i-1))/prod(1:i); % Dimension after polynomial expansion
end

hidden_size = feature_size;
output_size = class_num; % Network still trains on all 5 classes

%%%%%%%%%%%%%%%%%%%%One-hot Encoding%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
onehot=zeros(total_num,class_num); % Initialize matrix for one-hot encoded labels

identity_matrix=eye(class_num);
for i=1:total_num
    onehot(i,:)=identity_matrix(original_Labels(i),:);
end
onehot=onehot';

% feature=Normalization(feature,3);

Total_acc=0; % Counter for total accuracy
Total_conf_acc=zeros(composite_class_num,composite_class_num); % Counter for total confusion matrix

% New variables for AUC and F1
Total_AUC = 0;
Total_F1 = 0;
Total_sensitivity = 0;
Total_specificity = 0;

for testingNum=1:testNum
    accuracy = zeros(1, k);
    confusionMatrix=zeros(composite_class_num,composite_class_num,k);
    AUC_values = zeros(1, k);
    F1_values = zeros(1, k);
    sensitivity_values = zeros(1, k);
    specificity_values = zeros(1, k);
    
    % Cross-validation implementation ensures balanced class distribution in each fold and randomizes the dataset
    cv = cvpartition(original_Labels', 'kfold', k); 
    for i=1:k
        res_Train_Data=feature(:,cv.training(i));
        res_Test_Data=feature(:,cv.test(i));

        % Store original test labels for later mapping to composite classes
        original_Test_Labels = original_Labels(cv.test(i));

        % Normalize training set
        Tmin=min(res_Train_Data,[],2);
        Tmax=max(res_Train_Data,[],2);
        res_Train_Data=(res_Train_Data-Tmin)./(Tmax-Tmin)*(1-shift_num)+shift_num;

        % Normalize test set using training set parameters
        res_Test_Data=(res_Test_Data-Tmin)./(Tmax-Tmin)*(1-shift_num)+shift_num;

        %%%%%%%%%%%%%%%%%%%%%%% Polynomial Expansion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        TrainData=gpuArray(Expan(res_Train_Data,Power));
        TestData=gpuArray(Expan(res_Test_Data,Power));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        TrainLabels = gpuArray(onehot(:,cv.training(i)));
        TrainLabels_valid=gpuArray(original_Labels(:,cv.training(i)));
        
        % Original test labels (1-5)
        TestLabels_original=gpuArray(original_Labels(:,cv.test(i)));
    
        TrainData_valid=TrainData; % For training set validation
        res_Train_Data_valid=res_Train_Data;
        
        % Initialize weight matrices
        % Initialization types: 1=Random, 2=Gaussian, 3=Xavier-logistic, 4=Xavier-tanh, 5=Random
        W1 = gpuArray(Initialization(hidden_size, expan_size,7));
        W2 = gpuArray(Initialization(output_size, hidden_size,2));

        train_accurancy=0;
        test_accurancy=0;
        
        epsilon=0.00001;  % Prevent division by zero
        % Initialize mean of gradients (first moment):
        GW2=zeros(size(W2)); 
        GW1=zeros(size(W1));
        % Initialize variance of gradients without subtracting mean (second moment):
        MW2=zeros(size(W2)); 
        MW1=zeros(size(W1));
        
        % Train the network
        for epoch = 1:numEpochs
            % Training on the training set
            % Forward propagation
            z1 = W1*TrainData;
            hidden = Activate(z1,Acti_type); % Activate is the activation function
            z2 = W2*(hidden+res_Train_Data);  % res_Train_Data is original data without polynomial expansion
            output = softmax(z2);
            
            % Calculate output layer error
            d_softmax = output - TrainLabels;
            % Derivatives for W2 and hidden layer:
            d_W2 = d_softmax * (hidden+res_Train_Data)'/train_num;
            d_hidden = W2' * d_softmax; % Derivative for hidden layer
            d_z1 = d_hidden .* Activate_grad(z1,Acti_type);
            % Calculate derivatives for W1 and batch data:
            d_W1 = d_z1 * TrainData'/train_num;
    
            % Update parameters
            % Gradient optimization methods: 1=Adjustable decay, 2=AdaGrad, 3=RMSprop, 
            % 4=Momentum, 5=AdaM, 6=AdaM_2, 7=AdaM_3
            [W1,GW1,MW1]=Gradient_renewal(7,W1,d_W1,GW1,MW1,lr,epoch);
            [W2,GW2,MW2]=Gradient_renewal(7,W2,d_W2,GW2,MW2,lr,epoch);
            
            if epoch==numEpochs
                % Testing on training set (using original labels)
                z1 = W1*TrainData_valid;
                hidden = Activate(z1,Acti_type);
                z2 = W2*(hidden+res_Train_Data_valid);
                output = softmax(z2);
                % Calculate accuracy and loss
                [~, predictedLabels] = max(output);
                train_accurancy = mean(predictedLabels == TrainLabels_valid);
                
                % Testing on test set
                z1 = W1*TestData;
                hidden = Activate(z1,Acti_type);
                z2 = W2*(hidden+res_Test_Data);
                output = softmax(z2);
                % Calculate original class predictions
                [~, predictedLabels_original] = max(output);
                
                % Convert original test labels and predictions to composite classes
                % Map both true labels and predicted labels to composite classes
                testLabels_composite = zeros(size(TestLabels_original));
                predictedLabels_composite = zeros(size(predictedLabels_original));
                
                % Map true labels to composite classes
                for c=1:length(composite_class1)
                    testLabels_composite(TestLabels_original == composite_class1(c)) = 1;
                    predictedLabels_composite(predictedLabels_original == composite_class1(c)) = 1;
                end
                
                for c=1:length(composite_class2)
                    testLabels_composite(TestLabels_original == composite_class2(c)) = 2;
                    predictedLabels_composite(predictedLabels_original == composite_class2(c)) = 2;
                end
                
                % Only consider samples that belong to either composite class 1 or 2
                valid_indices = (testLabels_composite == 1) | (testLabels_composite == 2);
                
                % Calculate accuracy based on composite classes
                test_accurancy = mean(predictedLabels_composite(valid_indices) == testLabels_composite(valid_indices));
                
                % Get raw probabilities for composite classes for AUC calculation
                testProbabilities = gather(output');
                
                % Sum probabilities for classes within each composite class
                compProb1 = zeros(size(testLabels_composite));
                compProb2 = zeros(size(testLabels_composite));
                
                for c=1:length(composite_class1)
                    compProb1 = compProb1 + testProbabilities(:,composite_class1(c));
                end
                
                for c=1:length(composite_class2)
                    compProb2 = compProb2 + testProbabilities(:,composite_class2(c));
                end
                
                % Normalize probabilities
                totalProb = compProb1 + compProb2;
                compProb1 = compProb1 ./ totalProb;
                compProb2 = compProb2 ./ totalProb;
                
                % Only use valid indices (samples that belong to either composite class)
                compProb2_valid = compProb2(valid_indices);
                testLabels_composite_valid = testLabels_composite(valid_indices);
                
                % Calculate ROC and AUC
                [X, Y, ~, AUC] = perfcurve(gather(testLabels_composite_valid), gather(compProb2_valid), 2);
                AUC_values(i) = AUC;
                
                % Calculate confusion matrix for this fold using composite classes
                CM = confusionmat(gather(testLabels_composite(valid_indices)), gather(predictedLabels_composite(valid_indices)));
                
                % Ensure confusion matrix is 2x2 (some folds might not have samples from all classes)
                if size(CM, 1) < 2 || size(CM, 2) < 2
                    CM_full = zeros(2, 2);
                    CM_full(1:size(CM,1), 1:size(CM,2)) = CM;
                    CM = CM_full;
                end
                
                confusionMatrix(:,:,i) = CM;
                
                % Calculate F1 score and other metrics
                TP = CM(1,1);
                FP = CM(1,2);
                FN = CM(2,1);
                TN = CM(2,2);
                
                % Calculate sensitivity and specificity
                sensitivity = TP / (TP + FN);
                specificity = TN / (TN + FP);
                sensitivity_values(i) = sensitivity;
                specificity_values(i) = specificity;
                
                % F1 score calculation
                precision = TP / (TP + FP);
                recall = sensitivity; % Recall is the same as sensitivity
                if (precision + recall) > 0
                    F1 = 2 * (precision * recall) / (precision + recall);
                else
                    F1 = 0;
                end
                F1_values(i) = F1;
                
                fprintf('No.%d, K=%d, Test accuracy:%.4f\n', ...
                    testingNum, i, test_accurancy*100);
            end
        end
        accuracy(i)=test_accurancy;
    end
    Total_acc=Total_acc+mean(accuracy);
    Total_conf_acc=Total_conf_acc+mean(confusionMatrix,3);
    Total_AUC = Total_AUC + mean(AUC_values);
    Total_F1 = Total_F1 + mean(F1_values);
    Total_sensitivity = Total_sensitivity + mean(sensitivity_values);
    Total_specificity = Total_specificity + mean(specificity_values);
end
time=toc(t0); % End timing

Average_accuracy=Total_acc*100/testNum; % Calculate average accuracy across all experiments
Average_AUC = Total_AUC/testNum; % Calculate average AUC
Average_F1 = Total_F1/testNum; % Calculate average F1
Average_sensitivity = Total_sensitivity*100/testNum; % Calculate average sensitivity
Average_specificity = Total_specificity*100/testNum; % Calculate average specificity

Total_conf_acc=Total_conf_acc/testNum; % Average confusion matrix

% Extract values from confusion matrix
TP = Total_conf_acc(1,1);
FP = Total_conf_acc(1,2);
FN = Total_conf_acc(2,1);
TN = Total_conf_acc(2,2);

% Calculate accuracy, sensitivity, and specificity
accuracy = (TP + TN) / (TP + FP + FN + TN);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);

% Calculate F1 score
precision = TP / (TP + FP);
recall = sensitivity; % Recall is the same as sensitivity
F1 = 2 * (precision * recall) / (precision + recall);

% Output results
fprintf('\n\t Accuracy: %.4f, Sensitivity: %.4f, Specificity: %.4f, F1: %.4f, AUC: %.4f, Average time: %.4f\n', ...
    accuracy*100, sensitivity*100, specificity*100, Average_F1, Average_AUC, time/(testNum*k));

% Display confusion matrix with class names
disp('Confusion Matrix (Average across all experiments):');
fprintf('True Class \\ Predicted Class | Composite [');
for c=1:length(composite_class1)
    if c > 1
        fprintf(',');
    end
    fprintf('%d', composite_class1(c));
end
fprintf('] | Composite [');
for c=1:length(composite_class2)
    if c > 1
        fprintf(',');
    end
    fprintf('%d', composite_class2(c));
end
fprintf('] |\n');

fprintf('Composite [');
for c=1:length(composite_class1)
    if c > 1
        fprintf(',');
    end
    fprintf('%d', composite_class1(c));
end
fprintf('] | %.4f | %.4f |\n', Total_conf_acc(1,1), Total_conf_acc(1,2));

fprintf('Composite [');
for c=1:length(composite_class2)
    if c > 1
        fprintf(',');
    end
    fprintf('%d', composite_class2(c));
end
fprintf('] | %.4f | %.4f |\n', Total_conf_acc(2,1), Total_conf_acc(2,2));