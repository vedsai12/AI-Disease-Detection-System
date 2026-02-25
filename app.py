from flask import Flask

app = Flask(__name__)
import joblib

model = joblib.load("model.pkl")

print("Enter symptoms (1 = Yes, 0 = No)")

fever = int(input("Fever: "))
cough = int(input("Cough: "))
fatigue = int(input("Fatigue: "))

prediction = model.predict([[fever, cough, fatigue]])

if prediction[0] == 1:
    print("Disease Detected")
else:
    print("No Disease")
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)