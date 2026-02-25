from flask import Flask, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return """
    <h2>Disease Prediction</h2>
    <form action="/predict" method="post">
        Fever (0 or 1): <input type="text" name="fever"><br><br>
        Cough (0 or 1): <input type="text" name="cough"><br><br>
        Fatigue (0 or 1): <input type="text" name="fatigue"><br><br>
        <input type="submit" value="Predict">
    </form>
    """

@app.route("/predict", methods=["POST"])
def predict():
    fever = int(request.form["fever"])
    cough = int(request.form["cough"])
    fatigue = int(request.form["fatigue"])

    prediction = model.predict([[fever, cough, fatigue]])

    if prediction[0] == 1:
        return "<h3>Disease Detected</h3>"
    else:
        return "<h3>No Disease</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)