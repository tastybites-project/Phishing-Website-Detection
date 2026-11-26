# Phishing Website Detection Using Machine Learning

## 1. Project Overview

Phishing Website Detection is a Machine Learning-based project developed to identify potentially phishing websites and distinguish them from legitimate websites.

The system uses website characteristics and URL-related features to predict whether a website is phishing or legitimate.

The project implements three Machine Learning algorithms:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The best-performing model is selected using GridSearchCV and 5-Fold Cross Validation.

A Streamlit web application is developed to provide an interactive prediction interface.

## 2. Project Objectives

- To understand and analyze the phishing website dataset.
- To perform Exploratory Data Analysis (EDA).
- To clean and preprocess the dataset.
- To train multiple Machine Learning classifiers.
- To compare model performance.
- To apply 5-Fold Cross Validation.
- To optimize model parameters using GridSearchCV.
- To select the best model.
- To develop a web application for phishing prediction.

## 3. Dataset Description

**Dataset:** UCI Phishing Websites Dataset

**Source:** https://archive.ics.uci.edu/dataset/327/phishing+websites

Dataset information:
- Records: 11,055
- Input features: 30
- Target column: Result
- Target values:
  - -1 = Phishing
  - +1 = Legitimate

The dataset contains website characteristics related to URLs, domain information, HTTPS, and other website security indicators.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model saving and loading |
| Streamlit | Web application |
| VS Code | Development environment |

## 5. Machine Learning Algorithms

### Logistic Regression
A classification algorithm used to predict whether a website is phishing or legitimate.

### Decision Tree Classifier
A tree-based classification algorithm that makes predictions using decision rules.

### Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

## 6. Project Methodology

1. Dataset loading and understanding.
2. Exploratory Data Analysis.
3. Data cleaning and preprocessing.
4. Splitting the dataset into 80% training and 20% testing.
5. Training three Machine Learning models.
6. Evaluating models using Accuracy, Precision, Recall and F1-Score.
7. Applying 5-Fold Cross Validation.
8. Hyperparameter tuning using GridSearchCV.
9. Selecting the best model.
10. Final evaluation using the test dataset.
11. Predicting new website feature data.
12. Developing a Streamlit web application.

## 7. Model Evaluation

The following evaluation metrics are used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Phishing Recall is given particular attention because failing to identify a phishing website can create security risks.

**Actual evaluation results:** Refer to `reports/model_evaluation_results.csv`, `reports/cross_validation_results.csv`, and `reports/final_model_results.csv`.

## 8. Project Folder Structure

```text
Phishing-Website-Detection/
│
├── dataset/
│   ├── Training Dataset.arff
│   ├── phishing_websites.csv
│   ├── X_features.csv
│   ├── y_target.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── src/
│   ├── convert_dataset.py
│   ├── load_data.py
│   ├── eda.py
│   ├── prepare_data.py
│   ├── split_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── cross_validation.py
│   ├── hyperparameter_tuning.py
│   ├── final_evaluation.py
│   └── predict_new_website.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── best_phishing_model.pkl
│
├── reports/
│
├── app.py
├── requirements.txt
└── README.md
```

Additional tuned model files and generated reports are also stored in their respective folders.

## 9. Installation

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Phishing-Website-Detection
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## 10. Run the Project

To run the Streamlit web application:

```bash
streamlit run app.py
```

The application will generally open at:

```text
http://localhost:8501
```

## 11. Prediction System

The application accepts the 30 website feature values used during model training.

It displays:
- Predicted website category.
- Phishing probability, when supported by the selected model.

**Limitation:** The current application uses manually supplied feature values. It does not automatically extract all 30 features from a URL.

## 12. Conclusion

This project demonstrates how Machine Learning techniques can be used to classify potentially phishing and legitimate websites.

Three classification algorithms are trained and compared. Cross Validation and GridSearchCV are applied to improve model selection and parameter tuning.

The selected model is integrated into a Streamlit web application to provide an interactive prediction interface.

The system is an academic demonstration and should not be treated as a guarantee that a website is safe.

## 13. Future Scope

- Automatic extraction of website features from URLs.
- Real-time phishing website detection.
- Integration with browser extensions.
- Improved model performance using additional datasets.
- Deployment as an online web application.

## 14. Project Type

MCA Final Year Project – PBL

**Project Title:** Phishing Website Detection Using Machine Learning
