# 📊 MTN Customer Churn Prediction Web App

This project is a machine learning powered web application that predicts **customer churn for MTN users** based on customer behavior patterns such as data usage, device type, monthly spending, age, and satisfaction score.

The app uses:
- **Flask** (backend API)
- **HTML + CSS (MTN themed UI)** (frontend)
- **XGBoost / RandomForest / or chosen ML model**
- **Label Encoding + Standard Scaling** (for preprocessing)
- **Joblib** (to load model + preprocessing objects)

---

## 🚀 Live Demo
🔗 *Coming soon when deployed on Render*

---

## ✅ Features

- Web-based input form with MTN branding UI
- Real-time churn prediction using an ML model
- Preprocessing with LabelEncoder & StandardScaler
- JSON API endpoint (`/predict`)
- Mobile responsive frontend design

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**
   - Feature encoding using `LabelEncoder` for categorical columns
   - Standardization using `StandardScaler`

2. **Model Training**
   - Final optimized model saved as:
     ```
     xg_churn_model.pkl
     scaler.pkl
     encoder.pkl
     ```

3. **Prediction API**
   - Accepts JSON input with customer features
   - Returns churn prediction:  
     - `1 = Customer is likely to churn`
     - `0 = Customer will stay`

---

## 🗂 Project Structure

```
mtn-churn-app/
│ app.py
│ requirements.txt
│ Procfile
│ xg_churn_model.pkl
│ encoder.pkl
│ scaler.pkl
│
├── templates/
│     index.html
│
└── static/
      style.css
```

---

## 🔧 Installation & Running Locally

Clone the repo:

```sh
git clone https://github.com/yourusername/mtn-churn-app.git
cd mtn-churn-app
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Run app locally:

```sh
python app.py
```

The app will start on:

```
http://127.0.0.1:5000/
```

---

## ☁️ Deployment on Render

Upload everything to GitHub first.

### Required Files:
| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies |
| `Procfile` | Tells Render how to start the app |
| `app.py` | Main Flask application |
| `.pkl files` | Model and preprocessing |

### Procfile content

```
web: gunicorn app:app
```

---

## 📬 API Usage

Endpoint:

```
POST /predict
Content-Type: application/json
```

Request example:

```json
{
  "features": [35, "Jan-25", "Lagos", "iPhone", "Male", 4, "Good service", 12, "Premium", 25000, 3, 75000, 10.4]
}
```

Response:

```json
{
  "prediction": 1
}
```

---

## 🛠 Tech Stack

| Component | Technology |
|----------|------------|
| Backend | Flask |
| Machine Learning | Scikit-Learn / XGBoost |
| Preprocessing | StandardScaler + LabelEncoder |
| Deployment | Render |
| Frontend | HTML, CSS, JS |

---

## 📞 Contact / Support

If you want to collaborate or need help deploying on Render, feel free to reach out.

---

**Made with ❤️ by Olamide**
