# Scania Truck APS Failure Prediction: A Cost-Sensitive Classification Analysis

## 1. Project Overview

This project addresses the "APS Failure at Scania Trucks" challenge, which involves analyzing operational data to predict failures in a truck's Air Pressure System (APS). The primary goal is not just to build an accurate classifier, but to build one that is optimized for a specific, asymmetric business cost metric. The challenge emphasizes the high cost of missing a component failure (a False Negative) compared to the lower cost of a false alarm (a False Positive).

This repository contains the Jupyter Notebook and related files for the analysis and modeling performed for Section 1 of the Boost Credit Technical Assessment.

## 2. The Core Challenge

This problem is distinct from standard classification tasks due to two main factors:
* **Extreme Class Imbalance:** The training dataset contains 59,000 negative samples (no APS failure) and only 1,000 positive samples (APS failure), a ratio of 59:1.
* **Cost-Sensitive Metric:** The model's performance is evaluated using the formula: `Total Cost = (10 * Number of False Positives) + (500 * Number of False Negatives)`. This metric forces the model to heavily prioritize the recall of the positive class.

## 3. Methodology

The analysis followed a structured data science workflow, documented within the main Jupyter Notebook:

1.  **Exploratory Data Analysis (EDA):**
    * Identified and quantified the severe class imbalance.
    * Analyzed the widespread missing values across the 171 anonymized features.
    * Visualized feature distributions (e.g., `cs_005`, `bb_000`) to confirm their predictive power by observing different patterns between the positive and negative classes.

2.  **Data Preprocessing:**
    * Developed a robust preprocessing pipeline to handle missing data using median imputation.
    * Applied feature scaling using `StandardScaler` to normalize the feature space, which is essential for linear models like Logistic Regression.

3.  **Modeling and Hyperparameter Tuning:**
    * Implemented a custom scoring function in `scikit-learn` to directly optimize for the business cost metric during model tuning.
    * Employed `GridSearchCV` with `StratifiedKFold` cross-validation to find the optimal hyperparameters for several models.
    * Trained and evaluated multiple models, including **Random Forest**, **XGBoost**, and **LightGBM**, all configured with parameters (e.g., `scale_pos_weight`) to handle the class imbalance.

4.  **Ensemble Methods:**
    * To improve performance and robustness, hybrid models were created.
    * A **Voting Classifier** was implemented to average the predictions of the best-performing base models.
    * A **Stacking Classifier** was implemented to train a final meta-model on the predictions of the base models, learning the optimal way to combine their outputs.

## 4. How to Run This Project

### Prerequisites
* Python 3.8+
* Jupyter Notebook or JupyterLab

### Installation
1.  Clone this repository to your local machine.
2.  Install the required Python libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place the dataset files (`aps_failure_training_set.csv` and `aps_failure_test_set.csv`) in the root directory of the project.

### Usage
1.  Launch Jupyter Notebook or JupyterLab.
2.  Open the main notebook (`Scania_APS_Failure_Analysis.ipynb`).
3.  Run the cells sequentially from top to bottom to reproduce the entire workflow, from EDA to final model evaluation.

## 5. Summary of Results

The experiment concluded that the ensemble methods provided the best performance. The final model rankings based on the cost metric on the unseen test set were:

| Model                          | Test Set Cost | False Positives (FP) | False Negatives (FN) |
| ------------------------------ | ------------- | -------------------- | -------------------- |
| **Stacking Classifier (2 models)** | **10770** | **827** | **5** |
| XGBoost                        | 10940         | 494                  | 12                   |
| Voting Classifier (2 models)   | 11060         | 556                  | 11                   |
| Random Forest                  | 11740         | 624                  | 11                   |
| LightGBM                       | 21770         | 127                  | 41                   |

The **Stacking Classifier**, which combined the strengths of Random Forest and XGBoost, achieved the lowest cost. It excelled by being the most effective at minimizing the highly penalized False Negatives, successfully identifying all but 5 of the true failures in the test set.

## 6. Alternative Approaches & Future Work

To further improve the model, the following approaches could be explored:
* **Data-Level Sampling:** Use techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic positive samples and train on a more balanced dataset.
* **Decision Threshold Tuning:** Instead of using the default 0.5 probability threshold, analyze the precision-recall curve to find an optimal threshold that minimizes the custom cost function.
* **Advanced Feature Engineering:** Create interaction and polynomial features from the most important base features to help the models capture more complex relationships.