
import pandas as pd
import joblib
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, recall_score


# ==========================================
# STEP 1: CREATE REQUIRED FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ==========================================
# STEP 2: LOAD TRAINING DATA
# ==========================================

X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv").squeeze()

print("Training dataset loaded successfully!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ==========================================
# STEP 3: DEFINE CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# STEP 4: DEFINE PHISHING RECALL SCORER
# ==========================================

phishing_recall_scorer = make_scorer(
    recall_score,
    pos_label=-1,
    zero_division=0
)


# ==========================================
# STEP 5: DEFINE MODELS AND PARAMETERS
# ==========================================

model_configs = {

    "Logistic Regression": {

        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                random_state=42
            ))
        ]),

        "params": {
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["liblinear", "lbfgs"]
        }
    },


    "Decision Tree": {

        "model": DecisionTreeClassifier(
            random_state=42
        ),

        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },


    "Random Forest": {

        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),

        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    }
}


# ==========================================
# STEP 6: PERFORM GRID SEARCH CV
# ==========================================

results = []

best_recall = -1
best_model_name = None
best_model_object = None

for name, config in model_configs.items():

    print("\n======================================")
    print("Hyperparameter Tuning:", name)
    print("======================================")

    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring=phishing_recall_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Train using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Save tuned model
    filename = name.lower().replace(" ", "_") + "_tuned.pkl"

    filepath = os.path.join("models", filename)

    joblib.dump(best_model, filepath)

    # Display results
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        "Best Phishing Recall:",
        round(grid_search.best_score_, 4)
    )

    print("Tuned Model Saved:", filepath)

    # Store results
    results.append({
        "Model": name,
        "Best Parameters": str(grid_search.best_params_),
        "Best CV Phishing Recall": grid_search.best_score_
    })

    # Identify best model across all three
    if grid_search.best_score_ > best_recall:

        best_recall = grid_search.best_score_
        best_model_name = name
        best_model_object = best_model


# ==========================================
# STEP 7: SAVE BEST MODEL
# ==========================================

if best_model_object is not None:

    joblib.dump(
        best_model_object,
        "models/best_phishing_model.pkl"
    )

    print("\n======================================")
    print("BEST MODEL SELECTED")
    print("======================================")

    print("Best Model:", best_model_name)

    print(
        "Best Phishing Recall:",
        round(best_recall, 4)
    )

    print("Saved as: models/best_phishing_model.pkl")


# ==========================================
# STEP 8: SAVE TUNING RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== HYPERPARAMETER TUNING SUMMARY ==========")

print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/hyperparameter_tuning_results.csv",
    index=False
)


# ==========================================
# STEP 9: COMPLETION MESSAGE
# ==========================================

print("\n======================================")
print("HYPERPARAMETER TUNING COMPLETED!")
print("======================================")

print("All tuned models saved successfully!")
print("Results saved in reports folder.")
