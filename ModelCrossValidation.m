function ModelCrossValidation(model, X, y, k)
% A function to evaluate the average of model's performance in all folds

    fprintf("\nTraining set Evaluation : ");

    % Calculate Loss obtained by the cross-validation model
    ce = kfoldLoss(model);
    fprintf("\n\nkfoldLoss = " + num2str(ce));
    
    % Calculate average of performance metrics in all folds

    % Define array to collect metrics in all folds
    accuracy = zeros(1, k);
    precision = zeros(1, k);
    recall = zeros(1, k);
    f1 = zeros(1, k);
    
    % Loop through each fold
    for i = 1:k

        % Get index for the current fold
        testidx = test(model.Partition, i);
    
        % Get y for the current fold
        y_predicted = predict(model.Trained{i}, X(testidx, :));
        y_test = y.credit_status(testidx);
    
        % Compute confusion matrix for the current fold
        cm = confusionmat(y_test, y_predicted);
        
        % Define TP, FN, FP, TN
        TP = cm(2, 2);
        FN = cm(2, 1);
        FP = cm(1, 2);
        TN = cm(1, 1);
    
        % Calculate scores for the current fold
        accuracy(i) = (TP + TN) / (TP + FN + FP + TN);
        precision(i) = TP / (TP + FP);
        recall(i) = TP / (TP + FN);
        f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    
    end
    
    
    % Display average scores
    fprintf("\n\nAverage Accuracy: " + num2str(mean(accuracy)));
    fprintf("\nAverage Precision: " + num2str(mean(precision)));
    fprintf("\nAverage Recall: " + num2str(mean(recall)));
    fprintf("\nAverage F1 Score: " + num2str(mean(f1)));
    fprintf("\n")

end