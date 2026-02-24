import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Sample dataset
data = {
    "fever": [1, 1, 0, 1, 0, 0, 1, 0],
    "cough": [1, 0, 1, 1, 0, 1, 0, 0],
    "fatigue": [1, 1, 0, 1, 0, 0, 1, 0],
    "disease": [1, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["fever", "cough", "fatigue"]]
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))

# Save model
joblib.dump(model, "model.pkl")