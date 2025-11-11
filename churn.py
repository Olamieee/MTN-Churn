from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("xg_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_NAMES = [
    "Date of Purchase",
    "Age",
    "State",
    "MTN Device",
    "Gender",
    "Satisfaction Rate",
    "Customer Review",
    "Customer Tenure in months",
    "Subscription Plan",
    "Unit Price",
    "Number of Times Purchased",
    "Total Revenue",
    "Data Usage"
]

categorical_cols = ["Date of Purchase", "State", "MTN Device", "Gender", "Customer Review", "Subscription Plan"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data["features"]

    df = pd.DataFrame([features], columns=FEATURE_NAMES)

    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    df = pd.DataFrame(scaler.transform(df), columns=FEATURE_NAMES)

    prediction = model.predict(df)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)