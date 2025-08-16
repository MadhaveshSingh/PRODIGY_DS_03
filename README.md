# PRODIGY_DS_03

Decision Tree Classifier on Bank Marketing Dataset

This project implements a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit based on the Bank Marketing Dataset from the UCI Machine Learning Repository.

The dataset contains information about customers contacted through marketing campaigns, including demographic, social, and economic attributes. By applying machine learning techniques, we aim to identify patterns and key features influencing customer decisions.
🚀 Project Workflow

**1. Data Loading.**

The dataset is loaded from a CSV file (bank-additional.csv) using pandas.  
Since the dataset uses ; as a delimiter, it is handled accordingly.

**2. Data Preprocessing.**

All categorical variables are encoded using Label Encoding.  
Features (X) and target variable (y) are separated for training.

**3. Train-Test Split.**

The dataset is split into 80% training and 20% testing sets using train_test_split.

**4. Model Training.**

A Decision Tree Classifier is trained with:
Criterion: entropy  
Max Depth: 5  
Random State: 42

**5. Model Evaluation.**

Predictions are made on the test data.  
Performance is evaluated using:  
Accuracy Score  
Classification Report (Precision, Recall, F1-Score)

**6. Decision Tree Visualization.**

The trained decision tree is visualized using plot_tree() from scikit-learn.  
Features, class names, and decision paths are highlighted for better interpretability.

📊 Results

Accuracy: Printed after evaluation.  
Classification Report: Shows performance across classes.  
Decision Tree Plot: Displays decision rules, splits, and feature importance visually
