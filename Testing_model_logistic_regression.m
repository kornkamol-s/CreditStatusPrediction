%% Model Evaluation (1): Logistic Regresstion

%% Import train and test data
data = importdata('source\train_data.mat');
X_os_train = data.X_os_train;
y_os_train = data.y_os_train;

data = importdata('source\test_data.mat');
X_test = data.X_test;
y_test = data.y_test;

%% Evaluate model performance

% Define number of fold for cross validation
k = 5;

% Load final model
load('model\CV_Logistic_regression.mat');
load('model\Logistic_regression.mat');

% Training set using K-fold
ModelCrossValidation(CVLogit_optimal, X_os_train, y_os_train, k);

% Final testing set
ModelFinalValidation(Logit_optimal, X_test, y_test);