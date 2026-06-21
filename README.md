# 🩺 Diabetes Prediction System

A machine learning web app that predicts diabetes risk using **Logistic Regression** 
trained on the **PIMA Indians Diabetes Dataset**, served via a clean **Streamlit** UI.

---

## 📁 Project Structure

```
Diabetes_Prediction/
├── dataset/
│   └── Diabetes_prediction.csv     # PIMA Indians dataset (768 rows)
├── model/
│   ├── diabetes_model.pkl          # Trained Logistic Regression model
│   └── scaler.pkl                  # Fitted StandardScaler
├── train.py                        # Training pipeline
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🧠 Model Details

| Property         | Value                              |
|------------------|------------------------------------|
| Algorithm        | Logistic Regression (lbfgs)        |
| Dataset          | PIMA Indians Diabetes (768 rows)   |
| Train/Test Split | 80% / 20% stratified               |
| Preprocessing    | Zero-value imputation + StandardScaler |
| Test Accuracy    | ~71.4%                             |
| ROC-AUC Score    | ~0.823                             |

**Note:** The Kaggle dataset `mrsimple07/diabetes-prediction` is synthetically 
generated with near-random values (feature correlation ≈ 0), making it 
non-learnable. This project uses the authentic PIMA Indians dataset with 
the same column names but renamed target column `Outcome` → `Diagnosis`.

---

## ⚙️ Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Train the model
```bash
python train.py
```

Output:
- Prints accuracy, ROC-AUC, confusion matrix, classification report
- Saves `model/diabetes_model.pkl` and `model/scaler.pkl`

### Step 2 — Launch the app
```bash
streamlit run app.py
```

Opens at → **http://localhost:8501**

> ✅ The `.pkl` files are pre-included, so you can skip `train.py` and run the app directly.

---

## 🖥️ App Features

- 8 clinical input fields (Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, DPF, Age)
- **Run Prediction** button
- Result: **Diabetic** or **Non-Diabetic** with color-coded card
- **Confidence scores** with animated progress bars
- **Normal reference ranges** panel

---

## ⚠️ Disclaimer

For **educational and research purposes only**. Not a substitute for medical advice.

---

## 🛠️ Tech Stack

`Python 3.9+` · `Scikit-learn` · `Pandas` · `NumPy` · `Streamlit` · `Pickle`
