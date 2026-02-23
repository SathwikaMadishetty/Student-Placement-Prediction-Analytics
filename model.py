import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("data/placement.csv")

# Convert categorical Branch to dummy variables
data = pd.get_dummies(data, columns=["Branch"], drop_first=True)

# Features & Labels
X = data.drop("Placed", axis=1)
y = data["Placed"]

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))