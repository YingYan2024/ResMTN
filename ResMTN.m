clc;clear;

t0=tic; % Timing the entire process

% Set learning rate and training epochs
testNum=5; % Number of repeated experiments
numEpochs = 1000; % Maximum iterations for each fold in cross-validation
lr = 0.001; % Initial learning rate
k=10;  % 10-fold cross-validation
Acti_type = 4; % Activation function type: 1=RELU, 2=LeakyRELU, 3=Logistic, 4=tanh
Power=2; % Set the highest expansion order for MTN
class_num=2; % Number of classes
shift_num=0.2; % Shift parameter for normalization

%%%%%%%%%%%%%%%%%%% Data Import and Wavelet Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Data.mat
format long

% Data=round(Data);

Data=[Class_D Class_E];

total_num=size(Data,2); % Total number of samples
len_data=size(Data,1); % Length of each data sample

mother_wavelet='coif3'; % Type of mother wavelet
level=4; % Decomposition level
Wavelet=cell(1,level);
%Wavelet=cell(1,level+1);

% Performing wavelet decomposition for each sample
for i = 1:total_num
    if i==1
        [a,l]=wavedec(Data(:,i),level,mother_wavelet);
        [Wavelet{1},Wavelet{2},Wavelet{3},Wavelet{4}]=detcoef(a,l,1:level);
        %Wavelet{5}=appcoef(a,l,mother_wavelet);
        Wavelet{1}=zeros(size(Wavelet{1},1),total_num);
        Wavelet{2}=zeros(size(Wavelet{2},1),total_num);
        Wavelet{3}=zeros(size(Wavelet{3},1),total_num);
        Wavelet{4}=zeros(size(Wavelet{4},1),total_num);
    end
    [a,l]=wavedec(Data(:,i),level,mother_wavelet);
    [Wavelet{1}(:,i),Wavelet{2}(:,i),Wavelet{3}(:,i),Wavelet{4}(:,i)]=detcoef(a,l,1:level);
    %Wavelet{5}(:,i)=appcoef(a,l,mother_wavelet);
end

%%%%%%%%%%%%%%%%%%% Feature Extraction and Feature Selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Feature Extraction %%%%%%%%%%%%%%%%%%%%%%%%%
Max=zeros(level,total_num); % Maximum value
Min=zeros(level,total_num); % Minimum value
Std=zeros(level,total_num); % Standard deviation
Absaver=zeros(level,total_num); % Absolute mean
Energy=zeros(level,total_num); % Energy
Nstd=zeros(level,total_num); % Normalized standard deviation (easily affected by outliers)
Nenergy=zeros(level,total_num); % Normalized energy (not rigorous)
FuzApEn=zeros(level,total_num); % Fuzzy approximate entropy

%for i=1:level+1
for i=1:level
    Max(i,:)=max(Wavelet{i});
    Min(i,:)=min(Wavelet{i});
    Std(i,:)=std(Wavelet{i});
    Absaver(i,:)=mean(abs(Wavelet{i}));
    Energy(i,:)=sum(power(abs(Wavelet{i}),2),1);
    Nstd(i,:)=Std(i,:)/(max(Std(i,:))-min(Std(i,:)));
    Nenergy(i,:)=sqrt(Energy(i,:)/len_data);
    FuzApEn(i,:)=fuzzy_approx_entropy(Wavelet{i});
end

feature=gpuArray([Max;Min;Std;Absaver;Energy;Nstd;Nenergy;FuzApEn]);

% Calculate min and max values for each feature
%min_values = min(feature, [], 2);
%max_values = max(feature, [], 2);
% Perform normalization
%feature = (feature - min_values) ./ (max_values - min_values);
%%%%%%%%%%%%%%%%%%% Feature Selection and Arrangement %%%%%%%%%%%%%%%%%%%%%
[feature,sorted_indices]=feature_selection(feature,Labels',24,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform de-normalization
%feature = feature .* (max_values - min_values) + min_values;

train_num=round(total_num*(k-1)/k); % Size of training set (nine folds)

% Set network structure
feature_size=size(feature,1); % Dimension of feature vector
%expan_size=length(Taylor_expan5(feature(:,1),Power)); % Dimension after polynomial expansion

% Calculate expanded feature size based on polynomial order
expan_size=1;
for i=1:Power
    expan_size=expan_size + prod(feature_size:(feature_size+i-1))/prod(1:i); % Dimension after polynomial expansion
end

hidden_size = feature_size;
output_size = class_num;

%%%%%%%%%%%%%%%%%%%%One-hot Encoding%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
onehot=zeros(train_num,class_num); % Initialize matrix for one-hot encoded labels

identity_matrix=[1,0;0,1];
for i=1:total_num
    switch Labels(i)
        case 1
            onehot(i,:)=identity_matrix(1,:);
        case 2
            onehot(i,:)=identity_matrix(2,:);
    end
end % Add or reduce case statements based on number of classes
onehot=onehot';

% feature=Normalization(feature,3);

Total_acc=0; % Counter for total accuracy
Total_conf_acc=zeros(2,2); % Counter for total confusion matrix
for testingNum=1:testNum
    accuracy = zeros(1, k);
    confusionMatrix=zeros(class_num,class_num,k);

    % Cross-validation implementation ensures balanced class distribution in each fold and randomizes the dataset
    cv = cvpartition(Labels', 'kfold', k); 
    for i=1:k
        res_Train_Data=feature(:,cv.training(i));
        res_Test_Data=feature(:,cv.test(i));

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

%         TrainData=TrainData([2:end],:);
%         TestData=TestData([2:end],:);

        % plot(res_Train_Data(:,1));

        TrainLabels = gpuArray(onehot(:,cv.training(i)));
        TrainLabels_valid=gpuArray(Labels(:,cv.training(i)));
        TestLabels=gpuArray(Labels(:,cv.test(i)));
    
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
                % Testing on training set
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
                % Calculate accuracy and loss
                [~, predictedLabels] = max(output);
                test_accurancy = mean(predictedLabels == TestLabels);
                fprintf('No.%d, K=%d, Train accurancy:%.4f, Test accurancy:%.4f\n',testingNum,i,train_accurancy*100,test_accurancy*100);
            end
        end
        accuracy(i)=test_accurancy;
        confusionMatrix(:,:,i) = confusionmat(TestLabels, predictedLabels);
    end
    Total_acc=Total_acc+mean(accuracy);
    Total_conf_acc=Total_conf_acc+mean(confusionMatrix,3);

    %%%%%%%%%%%%%%%%% Record and Calculate Results after First Run %%%%%%%%%%%%%%%%%%%%%%%%%%
%     if testingNum == 1
%         accur=zeros(10,1);
%         sensi=zeros(10,1);
%         speci=zeros(10,1);
%         for p=1:k
%             TP = confusionMatrix(1,1,p);
%             FP = confusionMatrix(1,2,p);
%             FN = confusionMatrix(2,1,p);
%             TN = confusionMatrix(2,2,p);
%             accur(p) = (TP + TN) / (TP + FP + FN + TN);
%             sensi(p) = TP / (TP + FN);
%             speci(p) = TN / (TN + FP);
%         end
%         results=[accur,sensi,speci];
%     end
%     keyboard;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
time=toc(t0); % End timing

Average_accuracy=Total_acc*100/testingNum; % Calculate average accuracy across all experiments

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

% Output results
fprintf('\n\t Accuracy: %.4f, Sensitivity: %.4f, Specificity: %.4f, Average time: %.4f\n', accuracy, sensitivity, specificity,time/(testNum*k));