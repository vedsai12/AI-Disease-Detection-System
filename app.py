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