
# STUDENT PERFORMANCE PREDICTOR
# Model Training and Algorithm Comparison

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier


# 1. Load dataset
if os.path.exists("data/student_performance_1000.csv"):
    df = pd.read_csv("data/student_performance_1000.csv")
elif os.path.exists("student_performance_1000.csv"):
    df = pd.read_csv("student_performance_1000.csv")
else:
    raise FileNotFoundError("Student dataset not found!")

print("Dataset loaded successfully!")
print("Total records:", len(df))


# 2. Data preprocessing
required_columns = [
    "study_hours",
    "attendance",
    "assignment_score",
    "internal_marks",
    "previous_marks",
    "pass_fail"
]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Dataset columns do not match the expected format.")

df = df.drop_duplicates()
df = df.dropna(subset=["pass_fail"])

features = [
    "study_hours",
    "attendance",
    "assignment_score",
    "internal_marks",
    "previous_marks"
]

X = df[features]
y = df["pass_fail"]


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Define algorithms
models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(C=3, kernel="rbf", probability=True)
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )
}


# 5. Train and evaluate algorithms
results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3)
    })

    trained_models[name] = model

    print(name, "training completed!")


# 6. Compare algorithms
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\nALGORITHM COMPARISON")
print(comparison.to_string(index=False))

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

comparison.to_csv(
    "reports/algorithm_comparison.csv",
    index=False
)


# 7. Select best algorithm
best_name = comparison.iloc[0]["Algorithm"]
best_model = trained_models[best_name]

print("\nBest Algorithm:", best_name)
print("Accuracy:", comparison.iloc[0]["Accuracy (%)"], "%")


# 8. Final SVM model training
svm_model = models["SVM"]

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

print("\nFINAL SVM MODEL RESULTS")

print(
    "SVM Accuracy:",
    round(accuracy_score(y_test, svm_predictions) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(classification_report(y_test, svm_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))


# 9. Save trained SVM model
joblib.dump(
    svm_model,
    "models/student_performance_svm.pkl"
)

# Also save at the project root for compatibility
joblib.dump(
    svm_model,
    "student_performance_svm.pkl"
)

print("\nSVM model saved successfully!")


# 10. Save final results
svm_results = pd.DataFrame([{
    "Algorithm": "SVM",
    "Accuracy (%)": round(
        accuracy_score(y_test, svm_predictions) * 100, 2
    ),
    "Precision": round(
        precision_score(
            y_test, svm_predictions,
            average="weighted", zero_division=0
        ), 3
    ),
    "Recall": round(
        recall_score(
            y_test, svm_predictions,
            average="weighted", zero_division=0
        ), 3
    ),
    "F1 Score": round(
        f1_score(
            y_test, svm_predictions,
            average="weighted", zero_division=0
        ), 3
    )
}])

svm_results.to_csv(
    "reports/svm_final_results.csv",
    index=False
)

print("\nAll training and evaluation tasks completed!")
  
