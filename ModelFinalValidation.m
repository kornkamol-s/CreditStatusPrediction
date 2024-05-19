function ModelFinalValidation(model, X, y)
% A function to evaluate final performance of model using Test set
    
    fprintf("\nTest set Evaluation : ");
    
    % Start clock, to evaluate time in each process 
    tic;

    % Predict unseen test set
    [label, score] = model.predict(X);
       
    % Stop clock, and get time in milliseconds
    time = toc;
    fprintf('\n\nTesting Time: %.2f milliseconds\n', time* 1000);


    % Convert numerical value to string, easier to interpret
    label_strings = {'Good Credit', 'Bad Credit'};
    label_str = label_strings(label);
    y_str = label_strings(y.credit_status);

    % Show Confusion Matrix
    figure;
    confusionchart(y_str, label_str);
    
    
    % Plot ROC curve 
    figure;
    rocObj = rocmetrics(y.credit_status, score, model.ClassNames);
    plot(rocObj,ClassNames=model.ClassNames(2))
   
        
    % Compute confusion matrix
    cm = confusionmat(y.credit_status, label);

    % Define TP, FN, FP, TN
    TP = cm(2, 2);
    FN = cm(2, 1);
    FP = cm(1, 2);
    TN = cm(1, 1);

    % Calculate scores for the current fold
    accuracy = (TP + TN) / (TP + FN + FP + TN);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * (precision * recall) / (precision + recall);
 
    % Display average scores
    fprintf("\nAccuracy: " + num2str(accuracy));
    fprintf("\nPrecision: " + num2str(precision));
    fprintf("\nRecall: " + num2str(recall));
    fprintf("\nF1 Score: " + num2str(f1));
    fprintf("\n")

end