"""
app.py — Diabetes Prediction System
Run: streamlit run app.py
"""

import os
import pickle
import numpy as np
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f1923 100%);
    min-height: 100vh;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Hero */
.hero { text-align: center; padding: 2.5rem 0 1rem; }
.hero-badge {
    display: inline-block;
    background: rgba(0,188,212,.12);
    border: 1px solid rgba(0,188,212,.35);
    color: #00bcd4;
    font-family: 'Space Grotesk', sans-serif;
    font-size: .7rem; font-weight: 600;
    letter-spacing: .18em; text-transform: uppercase;
    padding: .35rem 1rem; border-radius: 20px; margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem; font-weight: 700;
    color: #e8f4f8; line-height: 1.15; margin: 0 0 .5rem;
}
.hero-title span { color: #00bcd4; }
.hero-sub { font-size: .9rem; color: #7a9bb5; margin: 0; }

/* Stats bar */
.stats-bar {
    display: flex; justify-content: center; gap: 2.5rem;
    padding: 1.2rem 0 1.8rem;
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin-bottom: 1.8rem;
}
.stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #00bcd4; text-align: center;
}
.stat-lbl { font-size: .7rem; color: #5a7a95; text-transform: uppercase; letter-spacing: .1em; text-align: center; }

/* Section label */
.sec-lbl {
    font-family: 'Space Grotesk', sans-serif;
    font-size: .68rem; font-weight: 600;
    letter-spacing: .15em; text-transform: uppercase;
    color: #00bcd4; margin-bottom: .9rem;
}

/* Input card */
.input-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1rem;
}

/* Number inputs */
.stNumberInput > div > div > input {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #e8f4f8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .9rem !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #00bcd4 !important;
    box-shadow: 0 0 0 2px rgba(0,188,212,.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #a0bfd0 !important; font-size: .8rem !important; font-weight: 500 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00bcd4, #0097a7) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important;
    padding: .85rem 2rem !important; letter-spacing: .03em !important;
    margin-top: .5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #26c6da, #00acc1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(0,188,212,.35) !important;
}

/* Result cards */
.res-diabetic {
    background: linear-gradient(135deg, rgba(239,83,80,.12), rgba(183,28,28,.08));
    border: 1px solid rgba(239,83,80,.35);
    border-radius: 16px; padding: 1.8rem; text-align: center; margin-top: .8rem;
}
.res-safe {
    background: linear-gradient(135deg, rgba(0,188,212,.1), rgba(0,150,136,.08));
    border: 1px solid rgba(0,188,212,.3);
    border-radius: 16px; padding: 1.8rem; text-align: center; margin-top: .8rem;
}
.res-icon  { font-size: 2.8rem; margin-bottom: .4rem; }
.res-label { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; margin: .3rem 0; }
.res-diabetic .res-label { color: #ef5350; }
.res-safe     .res-label { color: #00bcd4; }
.res-msg { font-size: .85rem; color: #7a9bb5; margin: .5rem 0 0; line-height: 1.5; }

/* Confidence */
.conf-row { display: flex; justify-content: space-between; margin-bottom: .35rem; }
.conf-lbl { font-size: .76rem; color: #7a9bb5; font-weight: 500; }
.conf-pct { font-family: 'Space Grotesk', sans-serif; font-size: .85rem; font-weight: 700; color: #e8f4f8; }
.bar-bg   { background: rgba(255,255,255,.06); border-radius: 99px; height: 8px; overflow: hidden; margin-bottom: .55rem; }
.bar-risk { height:100%; border-radius:99px; background: linear-gradient(90deg,#ef5350,#e53935); }
.bar-safe { height:100%; border-radius:99px; background: linear-gradient(90deg,#00bcd4,#00acc1); }

/* Empty state */
.empty-state {
    background: rgba(255,255,255,.02);
    border: 1px dashed rgba(255,255,255,.1);
    border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-bottom: 1rem;
}
.empty-icon  { font-size: 2.5rem; margin-bottom: .8rem; }
.empty-text  { font-family: 'Space Grotesk', sans-serif; color: #4a6a85; font-size: .9rem; font-weight: 500; }
.empty-text strong { color: #00bcd4; }

/* Info panel */
.info-panel {
    background: rgba(0,188,212,.04);
    border: 1px solid rgba(0,188,212,.14);
    border-radius: 12px; padding: 1.1rem 1.4rem; margin-top: 1.2rem;
}
.info-panel p { color: #7a9bb5; font-size: .78rem; margin: 0; line-height: 1.6; }
.info-panel strong { color: #00bcd4; }

/* Reference cards */
.ref-card {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .6rem;
}
.ref-name  { font-size: .75rem; font-weight: 600; color: #a0bfd0; margin-bottom: .15rem; }
.ref-range { font-family: 'Space Grotesk', sans-serif; font-size: .82rem; color: #00bcd4; }

hr { border: none; border-top: 1px solid rgba(255,255,255,.06); margin: 1.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    mp   = os.path.join(base, "model", "diabetes_model.pkl")
    sp   = os.path.join(base, "model", "scaler.pkl")
    if not os.path.exists(mp) or not os.path.exists(sp):
        st.error("⚠️  Model not found. Run `python train.py` first.")
        st.stop()
    with open(mp, "rb") as f: model  = pickle.load(f)
    with open(sp, "rb") as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🩺 AI-Powered Clinical Tool</div>
    <h1 class="hero-title">Diabetes <span>Prediction</span> System</h1>
    <p class="hero-sub">Logistic Regression · PIMA Indians Dataset · Scikit-learn</p>
</div>
<div class="stats-bar">
    <div><div class="stat-val">768</div><div class="stat-lbl">Patient Records</div></div>
    <div><div class="stat-val">8</div><div class="stat-lbl">Clinical Features</div></div>
    <div><div class="stat-val">71.4%</div><div class="stat-lbl">Test Accuracy</div></div>
    <div><div class="stat-val">0.823</div><div class="stat-lbl">ROC-AUC Score</div></div>
</div>
""", unsafe_allow_html=True)


# ─── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="sec-lbl">Patient Clinical Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pregnancies     = st.number_input("Pregnancies",                    min_value=0,   max_value=20,  value=3,    step=1)
        blood_pressure  = st.number_input("Blood Pressure (mm Hg)",         min_value=0,   max_value=200, value=72,   step=1)
        insulin         = st.number_input("Insulin (μU/mL)",                min_value=0,   max_value=900, value=79,   step=1)
        dpf             = st.number_input("Diabetes Pedigree Function",      min_value=0.0, max_value=3.0, value=0.47, step=0.01, format="%.3f")
    with c2:
        glucose         = st.number_input("Glucose (mg/dL)",                min_value=0,   max_value=300, value=117,  step=1)
        skin_thickness  = st.number_input("Skin Thickness (mm)",            min_value=0,   max_value=100, value=23,   step=1)
        bmi             = st.number_input("BMI (kg/m²)",                    min_value=0.0, max_value=70.0,value=32.0, step=0.1, format="%.1f")
        age             = st.number_input("Age (years)",                    min_value=1,   max_value=120, value=29,   step=1)

    st.markdown('</div>', unsafe_allow_html=True)
    clicked = st.button("🔍  Run Prediction", use_container_width=True)
    st.markdown("""
    <div class="info-panel">
        <p><strong>Clinical Disclaimer:</strong> This tool is for educational purposes only.
        Results <strong>do not replace</strong> professional medical diagnosis.
        Always consult a qualified healthcare provider.</p>
    </div>""", unsafe_allow_html=True)


with right:
    st.markdown('<div class="sec-lbl">Prediction Result</div>', unsafe_allow_html=True)

    if clicked:
        features        = np.array([[pregnancies, glucose, blood_pressure,
                                      skin_thickness, insulin, bmi, dpf, age]])
        features_scaled = scaler.transform(features)
        pred            = model.predict(features_scaled)[0]
        proba           = model.predict_proba(features_scaled)[0]
        p_diabetic      = proba[1] * 100
        p_safe          = proba[0] * 100

        if pred == 1:
            st.markdown(f"""
            <div class="res-diabetic">
                <div class="res-icon">⚠️</div>
                <div class="res-label">Diabetic</div>
                <p class="res-msg">Elevated diabetes risk detected based on the provided
                clinical markers. Please consult a healthcare professional.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-safe">
                <div class="res-icon">✅</div>
                <div class="res-label">Non-Diabetic</div>
                <p class="res-msg">Lower diabetes risk indicated. Maintain a healthy lifestyle
                and schedule routine health check-ups.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1.4rem;">
            <div class="sec-lbl">Confidence Scores</div>
            <div class="conf-row">
                <span class="conf-lbl">🔴 Diabetic Risk</span>
                <span class="conf-pct">{p_diabetic:.1f}%</span>
            </div>
            <div class="bar-bg"><div class="bar-risk" style="width:{p_diabetic:.1f}%"></div></div>
            <div class="conf-row" style="margin-top:.5rem;">
                <span class="conf-lbl">🔵 Non-Diabetic</span>
                <span class="conf-pct">{p_safe:.1f}%</span>
            </div>
            <div class="bar-bg"><div class="bar-safe" style="width:{p_safe:.1f}%"></div></div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-text">Enter patient data and click<br>
            <strong>Run Prediction</strong></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Normal Reference Ranges</div>', unsafe_allow_html=True)
    for name, rng in [
        ("Glucose",            "70 – 99 mg/dL (fasting)"),
        ("Blood Pressure",     "< 80 mm Hg (diastolic)"),
        ("BMI",                "18.5 – 24.9 kg/m²"),
        ("Insulin",            "2 – 25 μU/mL (fasting)"),
        ("Skin Thickness",     "~23 mm (avg. female)"),
        ("Pedigree Function",  "Lower = lower genetic risk"),
    ]:
        st.markdown(f"""
        <div class="ref-card">
            <div class="ref-name">{name}</div>
            <div class="ref-range">{rng}</div>
        </div>""", unsafe_allow_html=True)
