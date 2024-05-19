%% Clear session
clear; close all; clc;

%% Data Import

%%% Import file, and load data to independent, and dependent variables

% Import datafile
data = importdata('source\german.data');
X = data.textdata;
y = data.data;


% Convert arrays to tables, with headers
X = array2table(X, 'VariableNames', {'checking_account_status', 'duration_in_month', 'credit_history', 'purpose', 'credit_amount', ...
    'savings_account', 'employment_since', 'installment_rate', 'personal_status_sex', 'debtors_guarantors', ...
    'residence_since', 'property', 'age', 'other_installment_plans', 'housing', 'num_existing_credits', ...
    'job', 'num_people_maintenance', 'telephone', 'foreign_worker'});

y = array2table(y, 'VariableNames', {'credit_status'});


%% Data Cleaning

%%% Handle missing values, and convert datatype

% Check missing value
missing_X = ismissing(X);

if find(missing_X)
    fprintf(missing_X)
else
    fprintf("No Missing value in Input Features")
end


missing_y = ismissing(y);

if find(missing_y)
    fprintf('\n')
    fprintf(missing_y)
else
    fprintf("\nNo Missing value in Target")
end

% There is no missing data in this dataset


% Address type of features
category_columns = {'checking_account_status', 'credit_history', 'purpose', 'savings_account', ...
    'employment_since', 'personal_status_sex', 'debtors_guarantors', 'property', 'other_installment_plans', ...
    'housing', 'job', 'telephone', 'foreign_worker'};

numerical_columns = {'duration_in_month', 'credit_amount', 'installment_rate', 'residence_since', ...
    'age', 'num_existing_credits', 'num_people_maintenance'};


% Convert data type to double for numerical features
for col = numerical_columns
    X.(col{1}) = str2double(X.(col{1}));
end

% Convert data type to category for categorical features
for col = category_columns
    X.(col{1}) = categorical(X.(col{1}));
end


%% Data Exploration

data_table = [X, y];

% Visualise distribution of target column in pie chart
credit_catg = categorical(y.credit_status', [1 2], {'Good', 'Bad'});
figure;
piechart(credit_catg);
title("Distribution of Customer's Credit");


% Visualise numerical variables, using boxplot to see data characteristics
% of numerical features, including outliers, mean, and standard deviation
i = 1;
figure;

% Use subplot to plot all numerical variables characteistic
for col = numerical_columns
    subplot(2, 4, i);
    
    boxplot(X.(col{1}));
    title(col{1}, 'FontSize', 8, 'Interpreter', 'none');
    
    i = i + 1;
end


% Visualise categorical variables, using stacked bar chart in order to see
% ratio of customer's credit status in different category
i = 1;
figure;

for col = category_columns
    % Get all values for each categorical variable
    unique_vals = unique(data_table.(col{1}));

    % Define variables to count occurrences for each unique categorical value with credit_status 1 or 2
    count_1 = zeros(size(unique_vals));
    count_2 = zeros(size(unique_vals));

    % Count occurrences for each unique value in credit_status
    j = 1;
    for v = 1:length(unique_vals)
        index = data_table.(col{1}) == unique_vals(j);
        count_1(j) = sum(data_table.credit_status(index) == 1);
        count_2(j) = sum(data_table.credit_status(index) == 2);

        j = j + 1;
    end

    subplot(2, 7, i);
    bar(unique_vals, [count_1, count_2], 'stacked');
    xlabel(col{1}, 'FontSize', 8, 'Interpreter', 'none');
    ylabel('Count', 'FontSize', 8, 'Interpreter', 'none');

    i = i + 1;
end

% Add legend
Lgnd = legend('Good Credit', 'Bad Credit');
Lgnd.Position(1) = 0.78;
Lgnd.Position(2) = 0.3;


% Plotting the ksdensity to see difference of pattern between good, and bad credit
i = 1;
figure;

for col = numerical_columns
    subplot(2, 4, i);

    % Plot each feature with credit status = 1
    ksdensity(X.(col{1})(y.credit_status == 1));

    % To plot in the same figure
    hold on;

    % Plot each feature with credit status = 2
    ksdensity(X.(col{1})(y.credit_status == 2));

    % Finish plotting
    hold off;

    xlabel(col{1}, 'Interpreter', 'none');
    ylabel('Density');

    i = i + 1;
end

% Add legend
Lgnd = legend('Good Credit', 'Bad Credit');
Lgnd.Position(1) = 0.8;
Lgnd.Position(2) = 0.4;


%% Data Preprocessing

% Standardised numerical features, and encode categorical features

% Data Encoding for categorical features
for col = category_columns

    % Encode categorical features using ordinal encode
    X.(col{1}) = grp2idx(X.(col{1}));
end

% Encode target using ordinal encode
y.credit_status = grp2idx(y.credit_status);


% Data standardisation for numerical features
% Normalise numerical columns using zscore
X(:, numerical_columns) = array2table(zscore(table2array(X(:, numerical_columns))));


%% Feature Selection

% Calculate Correlation Coefficient for numerical features
numeric_table = [X(:, numerical_columns), y];

% Visualize Pearson correlation in heatmap
mat = corr(numeric_table{:, :});

figure;
heatmap(mat, 'XData', numeric_table.Properties.VariableNames, 'YData', ...
    numeric_table.Properties.VariableNames, 'Colormap', parula(5), 'Interpreter', 'none');
title('Correlation Matrix');


% We can see from the correlation matrix that credit_amount, and
% duration_in_month have strong correlation. So I decided to drop
% credit_amount out of features.

% Based on ksdensity and correlation matrix, we can see that residence_since, 
% num_existing_credits and num_people_maintenance
% has the same pattern between good, and bad credit, and they have very
% weak correlation with target labelled column. As of this reason, I
% decided to exclude these columns
numerical_columns = {'duration_in_month', 'installment_rate', 'age'};


% Feature ranking for categorical variables using Chi-Square test
[~, scores] = fscchi2(X(:, category_columns), y);
[scores_sort, idx] = sort(scores, 'descend');
category_columns_sort = category_columns(idx);

% Create a heatmap, showing feature ranking by Chi-Squared score
figure;
heatmap(scores_sort.', 'Colormap', parula, 'YDisplayLabels', category_columns_sort, 'Interpreter', 'none');
xlabel('Features');
ylabel('Score');
title('Chi-Square Feature Scores');

% As of score heatmap from chi-squre test, telephone, job, and
% debtors_guarantor have the lowest scores, showing that these features are
% not important to use.
category_columns = {'checking_account_status', 'credit_history', 'purpose', 'savings_account', ...
    'employment_since', 'personal_status_sex', 'property', 'other_installment_plans', ...
    'housing', 'foreign_worker'};

% Get final features for the model
X = X(:, [numerical_columns, category_columns]);

fprintf('\n\nFeatures: '+ strjoin(string(X.Properties.VariableNames), ', '));


%% Split data into Train and Test set

% Set Random Seed
rng(42);

% 25% testing set, 75% for training set
cv = cvpartition(size(X, 1), 'HoldOut', 0.25);

% Get index for test set
idx = cv.test;

X_train = X(~idx,:);
y_train = y(~idx,:);

X_test = X(idx,:);
y_test = y(idx,:);


%% Data Balancing

% Handle imbalanced data
fprintf("\n\nCredit Status Distribution : \n");
disp(tabulate(y_train.credit_status));

%This dataset shows imbalanced class with only 30% representing minority
%class

% Handle imbalanced data with simple oversampling, by duplicating the
% minority class's row into the original dataset.
X_oversampling = table2array(X_train(y_train.credit_status == 2, :));
y_oversampling = y_train(y_train.credit_status == 2, :);
X_os = [X_train; array2table(X_oversampling, 'VariableNames', X_train.Properties.VariableNames)];
y_os = [y_train; y_oversampling];

% Merge features with target
data_table = [X_os, y_os];

% Shuffle data before training
data_table = data_table(randperm(size(data_table, 1)), :);

% Extract the features (X) and target (y)
X_os_train = data_table(:, 1:end-1);
y_os_train = data_table(:, end);


% Show class distribution after oversampling
fprintf("\n\nCredit Status Distribution : \n");
disp(tabulate(y_os_train.credit_status));


%% Write data to text file
% These lines are commented out, to not replace everytime they were run

%save('source\train_data.mat', 'X_os_train', 'y_os_train');
%save('source\test_data.mat', 'X_test', 'y_test');
