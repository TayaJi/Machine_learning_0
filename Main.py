import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load dataset
data = pd.read_csv("Titanic-Dataset - Titanic-Dataset.csv")

print("Original dataset:")
print(data.head())


# 2. Remove unnecessary columns
data = data.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)


# 3. Separate input and output
X = data.drop("Survived", axis=1)
y = data["Survived"]


# 4. Select numerical and categorical columns
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


# 5. Clean numerical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# 6. Clean categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 7. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# 8. Create machine learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 9. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 10. Train the model
model.fit(X_train, y_train)


# 11. Test the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# 12. Predict a new passenger
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