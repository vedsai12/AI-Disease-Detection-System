from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Disease Prediction API Running"

@app.route('/predict')
def predict():
    fever = int(request.args.get("fever"))
    cough = int(request.args.get("cough"))
    fatigue = int(request.args.get("fatigue"))

    result = model.predict([[fever, cough, fatigue]])

    return "Disease Detected" if result[0] == 1 else "No Disease"

if __name__ == "__main__":
    app.run()