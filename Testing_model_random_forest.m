%% Model Evaluation (2): Random Forest

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

% Fiter features in testing set, according to feature importance step in
% training script, to be aligned with features in training data
selected_columns = {'duration_in_month', 'installment_rate', 'age', 'checking_account_status',  'credit_history', 'purpose', ...
    'savings_account', 'employment_since', 'personal_status_sex'};

X_test = X_test(:, selected_columns);

% Load final model
load('model\CV_Random_forest.mat');
load('model\Random_forest.mat');

% Training set using K-fold
ModelCrossValidation(cvrf, X_os_train, y_os_train, k);

% Final testing set
ModelFinalValidation(rf, X_test, y_test);