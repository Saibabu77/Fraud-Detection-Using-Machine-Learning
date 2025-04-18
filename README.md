# Fraud-Detection-Using-Machine-Learning
# Fraud Detection Pipeline Using Machine Learning

This project demonstrates an end-to-end machine learning pipeline for detecting fraudulent transactions using a synthetic dataset. The workflow includes preprocessing, exploratory analysis, model training, evaluation, and deployment.

## Dataset

- **File**: `synthetic_fraud_dataset.csv`
- **Target Column**: `Fraud_Label` (0 = Not Fraud, 1 = Fraud)
- **Additional Features**: Timestamps, transaction details, user-related data

## Workflow Overview

### 1. Data Preparation

The dataset is cleaned by:
- Removing high-cardinality columns such as `Transaction_ID` and `User_ID`
- Parsing the `Timestamp` to extract time-based features like hour and day of the week
- Handling missing values and encoding categorical variables using pipelines

### 2. Exploratory Data Analysis

Basic distributions and fraud frequency are visualized to understand the class imbalance. The class ratio shows a significant skew toward non-fraudulent transactions.

### 3. Modeling

Multiple classifiers are evaluated using SMOTE to address class imbalance:
- Logistic Regression
- Random Forest
- SVM
- XGBoost
- LightGBM
- CatBoost

Models are evaluated using AUC-ROC scores. While tree-based models like Random Forest and XGBoost show high scores, cross-validation is used to detect overfitting.

### 4. Model Evaluation

The best-performing model, Random Forest, is chosen for deployment. Evaluation metrics include:
- ROC Curve
- AUC Score
- Confusion Matrix
- Cross-validation results

Feature importance is also extracted to highlight key drivers of fraud detection.

### 5. Deployment

The final model is saved using `joblib` and applied to unseen samples. Each prediction is accompanied by a fraud probability score.

![image](https://github.com/user-attachments/assets/990d200d-777d-4ba3-8f76-c6efc977f055)

Summary of Prediction Results
The model was tested on five unseen transaction samples. Out of these:

Three transactions were confidently predicted as fraud with probabilities of 0.99, 0.98, and 0.99.

Two transactions were correctly identified as non-fraud with low probabilities of 0.02 and 0.01.

One transaction, with a probability of 0.06, was still classified as non-fraud, staying below the typical fraud threshold of 0.5.

Conclusion
The model demonstrates strong predictive performance on this small sample, accurately identifying both fraudulent and legitimate transactions. The high fraud probabilities show that the model responds well to risky patterns, while the low scores for non-fraud suggest low false positive risk. For deployment, further testing on a larger and more varied dataset is recommended, along with regular threshold evaluation to balance sensitivity and precision.




















## How to Run

1. Place `synthetic_fraud_dataset.csv` in the project folder
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost lightgbm catboost joblib
   ```
3. Run the Python script or notebook
4. The trained model will be saved as `random_forest_fraud_model.pkl`
5. Sample predictions will display fraud probabilities for unseen data.

## Notes

- High AUC scores do not guarantee real-world performance. The model should be tested on production-like data.
- Further tuning and cost-sensitive learning can improve fraud recall without increasing false positives.
