import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load the original dataset
data = pd.read_csv("Titanic-Dataset - Titanic-Dataset.csv")

print("Original dataset:")
print(data.head())


# 2. Remove unnecessary columns
data = data.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)


# 3. Clean missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])


# 4. Save the cleaned dataset
data.to_csv("Cleaned_Titanic.csv", index=False)

print("\nCleaned dataset saved as Cleaned_Titanic.csv")


# 5. Separate input and output
X = data.drop("Survived", axis=1)
y = data["Survived"]


# 6. Select numerical and categorical columns
numerical_columns = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_columns = [
    "Gender",
    "Embarked"
]


# 7. Prepare numerical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# 8. Prepare categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 9. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# 10. Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 11. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 12. Train model
model.fit(X_train, y_train)


# 13. Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# 14. Predict a new passenger
new_passenger = pd.DataFrame([{
    "Pclass": 1,
    "Gender": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80,
    "Embarked": "C"
}])

prediction = model.predict(new_passenger)


if prediction[0] == 1:
    print("Prediction: Passenger survived")
else:
    print("Prediction: Passenger did not survive")