import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load the Titanic dataset
data = pd.read_csv("Titanic-Dataset - Titanic-Dataset.csv")

print("===== TITANIC DATASET =====")

print("Dataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

# Analyze the target variable

print("\n===== SURVIVAL ANALYSIS =====")

print("Survival count:")
print(data["Survived"].value_counts())

print("\nSurvival percentage:")
print(
    (data["Survived"].value_counts(normalize=True) * 100).round(2)
)
print("\n===== SURVIVAL BY GENDER =====")

gender_survival = data.groupby("Gender")["Survived"].mean() * 100

print(gender_survival.round(2))

print("\n===== SURVIVAL BY CLASS =====")

class_survival = data.groupby("Pclass")["Survived"].mean() * 100

print(class_survival.round(2))

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

print("\n===== DATA CLEANING =====")

# Create FamilySize
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# Create IsAlone
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

print("\nNew features created:")
print(data[["SibSp", "Parch", "FamilySize", "IsAlone"]].head())

# Remove unnecessary columns
data = data.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)

print("\nColumns after removing unnecessary data:")
print(data.columns.tolist())

# Handle missing values

data["Age"] = data["Age"].fillna(
    data["Age"].median()
)

data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Save cleaned dataset

data.to_csv(
    "Cleaned_Titanic.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")

# --------------------------------------------------
# PREPARE DATA FOR MACHINE LEARNING
# --------------------------------------------------

X = data.drop("Survived", axis=1)
y = data["Survived"]

print("\n===== MACHINE LEARNING DATA =====")

print("Input features:")
print(X.columns.tolist())

print("\nTarget:")
print("Survived")

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN TEST SPLIT =====")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Define numerical and categorical features

numerical_columns = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone"
]

categorical_columns = [
    "Gender",
    "Embarked"
]

# Preprocessing for numerical data

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Preprocessing for categorical data

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])
# --------------------------------------------------
# LOGISTIC REGRESSION MODEL
# --------------------------------------------------

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions
logistic_pred = logistic_model.predict(X_test)

# Calculate accuracy
logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print("\n===== LOGISTIC REGRESSION =====")
print(
    "Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)
# --------------------------------------------------
# DECISION TREE MODEL
# --------------------------------------------------

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

# Train the model
decision_tree_model.fit(X_train, y_train)

# Make predictions
decision_tree_pred = decision_tree_model.predict(X_test)

# Calculate accuracy
decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

print("\n===== DECISION TREE =====")
print(
    "Accuracy:",
    round(decision_tree_accuracy * 100, 2),
    "%"
)
# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions
random_forest_pred = random_forest_model.predict(X_test)

# Calculate accuracy
random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

print("\n===== RANDOM FOREST =====")
print(
    "Accuracy:",
    round(random_forest_accuracy * 100, 2),
    "%"
)
# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

print("\n===== MODEL COMPARISON =====")

models = {
    "Logistic Regression": logistic_accuracy,
    "Decision Tree": decision_tree_accuracy,
    "Random Forest": random_forest_accuracy
}

for name, accuracy in models.items():
    print(
        name,
        ":",
        round(accuracy * 100, 2),
        "%"
    )
    # --------------------------------------------------
# SELECT BEST MODEL
# --------------------------------------------------

best_model_name = max(
    models,
    key=models.get
)

print("\nBest model:", best_model_name)
# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

if best_model_name == "Logistic Regression":
    best_prediction = logistic_pred
elif best_model_name == "Decision Tree":
    best_prediction = decision_tree_pred
else:
    best_prediction = random_forest_pred


print("\n===== BEST MODEL EVALUATION =====")

print(
    "Accuracy:",
    round(accuracy_score(y_test, best_prediction) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, best_prediction) * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall_score(y_test, best_prediction) * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1_score(y_test, best_prediction) * 100, 2),
    "%"
)
# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    best_prediction
)

print(cm)
# --------------------------------------------------
# MODEL COMPARISON GRAPH
# --------------------------------------------------

model_names = list(models.keys())
model_scores = [
    value * 100 for value in models.values()
]

plt.figure(figsize=(8, 5))

plt.bar(model_names, model_scores)

plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Titanic Model Accuracy Comparison")

plt.ylim(0, 100)

for i, score in enumerate(model_scores):
    plt.text(
        i,
        score + 1,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()
# --------------------------------------------------
# NEW PASSENGER PREDICTION
# --------------------------------------------------

new_passenger = pd.DataFrame([{
    "Pclass": 1,
    "Gender": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80,
    "Embarked": "C",
    "FamilySize": 1,
    "IsAlone": 1
}])

if best_model_name == "Logistic Regression":
    final_model = logistic_model

elif best_model_name == "Decision Tree":
    final_model = decision_tree_model

else:
    final_model = random_forest_model


prediction = final_model.predict(new_passenger)

print("\n===== NEW PASSENGER PREDICTION =====")

if prediction[0] == 1:
    print("Prediction: Passenger survived")
else:
    print("Prediction: Passenger did not survive")