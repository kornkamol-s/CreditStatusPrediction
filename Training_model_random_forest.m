%% Initialise Model (2): Random Forest

fprintf("\n\n-----------------------------------------------");
fprintf("\nRandom Forest");
fprintf("\n-----------------------------------------------");


%% Import train data
data = importdata('source\train_data.mat');
X_os_train = data.X_os_train;
y_os_train = data.y_os_train;


%% Feature Importance for Random Forest
% Reference from MathWorks Documentation: https://uk.mathworks.com/help/stats/select-predictors-for-random-forests.html
%----------------------------------------------


% Set random seed
rng(42);

% Initialise Random Forest model using ensemble classification model
% Use Reproducible to control randomness for each tree
% Use Bag method to randomly select predictors in each split
t = templateTree('Reproducible', true);
rf = fitcensemble(X_os_train, y_os_train, 'Method', 'Bag', 'Learners', t);

% Estimate predictor importance using permutation of OOB
% This method is to assess importance for each predictor, by compaing the
% importance, and mean square error gained
oob = oobPermutedPredictorImportance(rf);
mse_gain = predictorImportance(rf);

% Normalise values in order to compare in the same scale
normalized_oob = (oob - min(oob(:))) / (max(oob(:)) - min(oob(:)));
normalized_mse = (mse_gain - min(mse_gain(:))) / (max(mse_gain(:)) - min(mse_gain(:)));

% Plot Predictor Importance Estimation Comparison between OOB, and MSE method
figure;

plot(normalized_oob');
hold on;
plot(normalized_mse');
hold off;

title('Predictor Importance');
xlabel('Features');
ylabel('Importance');

xticks(1:numel(rf.PredictorNames));
xticklabels(rf.PredictorNames);
set(gca, 'TickLabelInterpreter', 'none');

legend('OOB', 'MSE');


% As seen in the importance chart, last 4 features (property, other_installment_plans, housing, foreign_worker) 
% seem to be less important. They will be removed from predictors
selected_columns = {'duration_in_month', 'installment_rate', 'age', 'checking_account_status',  'credit_history', 'purpose', ...
    'savings_account', 'employment_since', 'personal_status_sex'};

X_os_train = X_os_train(:, selected_columns);


%% Hyperparameter tuning
% Reference from MathWorks Documentation: 
% https://uk.mathworks.com/help/stats/fitcensemble.html
% https://uk.mathworks.com/help/stats/hyperparameters.html
% ----------------------------------------------


% Start clock, to evaluate time in each process 
tic;

% Set Random Seed
rng(42);

% Get list of parameter for fitcensemble with Tree learner
params = hyperparameters('fitcensemble', X_os_train, y_os_train, 'Tree');
for i = 1:length(params)
    disp(i), disp(params(i))
end

% Define range of each parameter, to have more control and reduce
% overfitting issue

% Method, "Bag" is chosen for method parameter
params(1).Optimize = false;

% LearnRate 
params(3).Optimize = false;

% NumLearningCycles
params(2).Range = [100,600];
params(2).Optimize = true;

% MinLeafSize
params(4).Range = [1,100];
params(4).Optimize = true;

% MaxNumSplits
params(5).Range = [1,100];
params(5).Optimize = true;

% NumVariablesToSample
params(7).Optimize = false;

% Set Random Seed
rng(42);

% Use Reproducible to control randomness for each tree
t = templateTree('Reproducible',true);

% Use automatic hyperparameter tuning with the range of selected
% parameters, and randomly select the combination up to 50 times, using
% k-fold cross validation
rf = fitcensemble(X_os_train, y_os_train, 'Method', 'Bag', 'OptimizeHyperparameters', params, 'Learners', t, ...
    'HyperparameterOptimizationOptions', struct("MaxObjectiveEvaluations" , 50, 'KFold', 5));

% Get best parameters
best_params = rf.HyperparameterOptimizationResults.XAtMinEstimatedObjective;
num_split = best_params.MaxNumSplits;
cycle = best_params.NumLearningCycles;
leaf = best_params.MinLeafSize;

% Stop clock, and get time in milliseconds
time = toc;
fprintf('\n\nParameter Tuning Time: %.2f milliseconds\n', time* 1000);


%% Train model using best parameters
% ----------------------------------------------

% Set random seed
rng(42);

% Start clock, to evaluate time in each process 
tic; 

% Use Reproducible to control randomness for each tree
% Provide the optimal value of MaxNumSplits, and MinLeafSize to limit the
% maximum number of branches, and the minimum number of leaf node in each
% tree
t = templateTree('Reproducible', true, 'MaxNumSplits', num_split, 'MinLeafSize', leaf);

% Train Random Forest model, using k-fold cross validation, to evaluate
% model performance in training set
% NumLearningCycles is provided to control number of trees in the forest
cvrf = fitcensemble(X_os_train, y_os_train, 'Method', 'Bag', 'KFold', 5, 'NumLearningCycles', cycle, 'Learners', t);

% Stop clock, and get time in milliseconds
time = toc;
fprintf('\n\nTraining Time: %.2f milliseconds\n', time* 1000);

% Train final model to assess model performance in unseen test set
rf = fitcensemble(X_os_train, y_os_train, 'Method', 'Bag', 'NumLearningCycles', cycle, 'Learners', t);
disp(cvrf);


%% Save Final models to be evaluated in Testing script
% These lines are commented out, to not replace the models everytime they
% were run, as hyperparameter tuning process are done with randomly select
% the combination of params. This will ensure the same performance metrics
% for every tests.

%save('model\Random_forest.mat', 'rf');
%save('model\CV_Random_forest.mat', 'cvrf');