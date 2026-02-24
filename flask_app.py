from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "AI Disease Detection API is Running 🚀"

@app.route("/predict", methods=["GET"])
def predict():
    try:
        fever = int(request.args.get("fever"))
        cough = int(request.args.get("cough"))
        fatigue = int(request.args.get("fatigue"))

        prediction = model.predict([[fever, cough, fatigue]])

        if prediction[0] == 1:
            result = "Disease Detected"
        else:
            result = "No Disease"

        return jsonify({
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "prediction": result
        })

    except:
        return jsonify({"error": "Please provide fever, cough, fatigue (0 or 1)"})


if __name__ == "__main__":
    app.run(debug=True)