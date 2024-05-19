%% Initialise Model (1): Logistic Regresstion

fprintf("\n\n-----------------------------------------------");
fprintf("\nLogictic Regression");
fprintf("\n-----------------------------------------------");


%% Import train data
data = importdata('source\train_data.mat');
X_os_train = data.X_os_train;
y_os_train = data.y_os_train;


%% Find Good Lasso Penalty Using Cross-Validation, 
% Reference from MathWorks Documentation: https://uk.mathworks.com/help/stats/fitclinear.html
%----------------------------------------------


% Start clock, to evaluate time in each process 
tic;

% Define range of lambda value, and gradient tolerance value
Lambda = logspace(-6, -0.5, 11);
GradientTolerance_values = 1e-10;

% Set Random Seed
rng(42);

% Construct Linear Regression Models, with 5-folds cross validation, Lasso
% Regularization and SpaRSA Method
% The models are built 11 times, with different value of Regularization
% strength
CVLogit = fitclinear(X_os_train, y_os_train, 'KFold', 5, 'Learner', 'logistic', ...
    'Solver', 'sparsa', 'Regularization', 'lasso', ...
    'Lambda', Lambda, 'GradientTolerance', GradientTolerance_values);

% Get classification errors, using to evaluate the optimal point of regularization strength
ce = kfoldLoss(CVLogit);

% Define the same Linear Regression Model without cross-validation to get
% the number of remaining predictors (features with non-zero coefficients),
% after Lasso Regularization
Logit = fitclinear(X_os_train, y_os_train, 'Learner', 'logistic', ...
    'Solver', 'sparsa', 'Regularization', 'lasso', ...
    'Lambda', Lambda, 'GradientTolerance', GradientTolerance_values);


% Stop clock, and get time in milliseconds
time = toc;
fprintf('\n\nParameter Tuning Time: %.2f milliseconds\n', time* 1000);

% Get number of the remaining predictors
non_zero_coef = sum(Logit.Beta ~= 0);

% Set value of -inf to 0, in order to visualize in graph
non_zero_coef(non_zero_coef==0) = 1;

%  Plot classification error and number of predictors for each strength on log scale
figure;
[ax, err, p] = plotyy(log10(Lambda), log10(ce), log10(Lambda), log10(non_zero_coef)); 

err.Marker = 'o';
p.Marker = 'o';

ylabel(ax(1), 'log_{10} classification error');
ylabel(ax(2), 'log_{10} nonzero-coefficient frequency');
xlabel('log_{10} Lambda');
title('Regularization Strength Tuning');


% After inspecting the visualisation, we can see that the optimal point is 10th 
% with low classification error, and less non-zero coefficients (Less complex model).
Logit_optimal = selectModels(Logit, 10);


% Display remaining predictors
idx = find(Logit_optimal.Beta ~= 0);
predictors = Logit_optimal.PredictorNames(idx);
num_of_features = num2str(size(idx, 1));

fprintf("\n\nNon-zero Beta coefficients: " + num_of_features + "\n Features : " + strjoin(predictors, ', '));


%% Train Logistic Regression model, with selected Regularization Strength

% Set Random Seed
rng(42);

% Start clock, to evaluate time in each process 
tic;

% Final model to evaluate on the training set using cross-validation to assess performance
CVLogit_optimal = fitclinear(X_os_train, y_os_train, 'KFold', 5, 'Learner', 'logistic', ...
    'Solver', 'sparsa', 'Regularization', 'lasso', ...
    'Lambda', Logit_optimal.Lambda, 'GradientTolerance', GradientTolerance_values);

% Stop clock, and get time in milliseconds
time = toc;
fprintf('\n\nTraining Time: %.2f milliseconds\n', time* 1000);

% Final model to evaluate on the testing set
Logit_optimal = fitclinear(X_os_train, y_os_train, 'Learner', 'logistic', ...
    'Solver', 'sparsa', 'Regularization', 'lasso', ...
    'Lambda', Logit_optimal.Lambda, 'GradientTolerance', GradientTolerance_values);

disp(CVLogit_optimal);


%% Save Final models to be evaluated in Testing script
% These lines are commented out, to not replace the models everytime they were run

%save('model\Logistic_regression.mat', 'Logit_optimal');
%save('model\CV_Logistic_regression.mat', 'CVLogit_optimal');