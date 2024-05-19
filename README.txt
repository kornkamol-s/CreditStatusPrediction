# ---------------------------------------------------------
# Project Name
# ---------------------------------------------------------
A comparison of Logistic Regression and Random Forest on Predicting German Credit Status


# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------
This project aims at predicting customer credit status using two different classification models to compare their performance, along with their advantages and disadvantages. The models are constructed with German Credit Data provided from UCI Machine Learning Repository.


# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------
To run test the models performance, execute these files;

Testing_model_logistic_regression.m
Testing_model_random_forest.m


# ---------------------------------------------------------
# Folder Structure
# ---------------------------------------------------------
LR-RF_Credit_status.zip/
|-- source/
|   |-- german.data
|   |-- train_data.mat
|   |-- test_data.mat
|-- model/
|   |-- CV_Logistic_regression.mat
|   |-- Logistic_regression.mat
|   |-- CV_Random_forest.mat
|   |-- Random_forest.mat
|-- Initial_data_analysis.m
|-- Training_model_logistic_regression.m
|-- Testing_model_logistic_regression.m
|-- Training_model_random_forest.m
|-- Testing_model_random_forest.m
|-- requirements.txt
|-- README.txt


# ---------------------------------------------------------
# Files Description
# ---------------------------------------------------------
1. source - A folder to store source data used for the models
- german.data - Orignal source data from provider, UCI Machine Learning Repository
- train_data.mat - Data that was cleaned and preprocessed, with normalization, encoding, oversampling, and feature selection, for training models.
- test_data.mat - Data that was cleaned and preprocessed, with normalization, encoding, and feature selection. This subset data remains unseen aimed at evaluating the performance and model generalization.

2. model - A folder to store the final models for both Logistic Regression and Random Forest
- CV_Logistic_regression.mat - Final cross-validated Logistic Regression model, to evaluate performance on training set
- Logistic_regression.mat - Final Logistic Regression model, to evaluate performance on testing set
- CV_Random_forest.mat - Final cross-validated Random Forest model, to evaluate performance on training set
- Random_forest.mat - Final Random Forest model, to evaluate performance on testing set

3. Initial_data_analysis.m - This MatLab file is for intial analysis, data cleaning, and data preprocessing (normalization, encoding, balancing, and predictor selection)

4. Training_model_logistic_regression.m - This MatLab file is for training Logistic Regression model

5. Testing_model_logistic_regression.m - This MatLab file is for final evaluation the performance of Logistic Regression

6. Training_model_random_forest.m - This MatLab file is for training Random Forest model

7. Testing_model_random_forest.m - This MatLab file is for final evaluation the performance of Random Forest

8. requirements.txt - A file contains MatLab Version and dependencies on Add-On ToolBoxes