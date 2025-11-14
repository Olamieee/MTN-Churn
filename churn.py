from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='.')

# Load model, encoder and scaler
model = joblib.load("xg_churn_model.pkl")
label_encoders = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
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

CATEGORICAL_COLS = [
    "Date of Purchase",
    "State",
    "MTN Device",
    "Gender",
    "Customer Review",
    "Subscription Plan"
]

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(os.path.dirname(__file__), "churn.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = data["features"]

    try:
        df = pd.DataFrame([features], columns=FEATURES)
    except Exception as e:
        return jsonify({"error": f"Column mismatch: {e}"}), 400

    # Encode categoricals safely
    for col in CATEGORICAL_COLS:
        encoder = label_encoders[col]
        try:
            df[col] = encoder.transform(df[col])
        except:
            df[col] = -1

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale
    try:
        scaled = scaler.transform(df)
    except Exception as e:
        return jsonify({"error": f"Scaler error: {e}"}), 500

    # Predict
    try:
        pred = int(model.predict(scaled)[0])
    except Exception as e:
        return jsonify({"error": f"Model error: {e}"}), 500

    return jsonify({"prediction": pred})


if __name__ == "__main__":
    app.run(debug=True)
