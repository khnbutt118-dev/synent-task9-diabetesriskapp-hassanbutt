"""Task 9 - Diabetes risk screening app (Streamlit). Educational demo only, not medical advice."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Diabetes Risk Screening Demo", page_icon="🩺")


@st.cache_resource
def load():
    model = joblib.load(BASE / "model" / "model.joblib")
    info = json.loads((BASE / "model" / "feature_info.json").read_text(encoding="utf-8"))
    return model, info


model, info = load()
st.title("🩺 Diabetes Risk Screening Demo")
st.caption("Enter patient measurements and get a model-based risk estimate. "
           "This is a data-science learning project and NOT a diagnostic tool.")

labels = {"Pregnancies": "Number of pregnancies", "Glucose": "Plasma glucose (mg/dL)",
          "BloodPressure": "Diastolic blood pressure (mm Hg)", "SkinThickness": "Skin-fold thickness (mm)",
          "Insulin": "2-hour serum insulin (mu U/ml)", "BMI": "Body mass index (kg/m2)",
          "DiabetesPedigreeFunction": "Diabetes pedigree function", "Age": "Age (years)"}
values = {}
cols = st.columns(2)
for i, f in enumerate(info["features"]):
    s = info["feature_stats"][f]
    with cols[i % 2]:
        step = 1.0 if (s["max"] - s["min"]) > 20 else 0.01
        values[f] = st.number_input(labels.get(f, f), min_value=float(s["min"]), max_value=float(s["max"]) * 1.5,
                                    value=float(s["median"]), step=step)

if st.button("Predict risk", type="primary"):
    X = pd.DataFrame([values], columns=info["features"])
    p = float(model.predict_proba(X)[0, 1])
    st.progress(min(max(p, 0.0), 1.0))
    st.metric("Estimated probability of diabetes", f"{p * 100:.1f}%")
    if p < 0.30:
        st.success("Lower estimated risk")
    elif p < 0.60:
        st.warning("Moderate estimated risk")
    else:
        st.error("Higher estimated risk: a clinical check would be advisable")
    st.caption("Model output only. Please consult a healthcare professional for diagnosis.")

with st.expander("About this model"):
    m = info["test_metrics"]
    st.write(f"Model: **{info['model_name']}**. Held-out test results: accuracy {m['accuracy']}, "
             f"recall {m['recall']}, precision {m['precision']}, F1 {m['f1']}, ROC-AUC {m['roc_auc']}.")
    st.write("Data: Pima Indians Diabetes Database (Kaggle / UCI). Trained with imputation, scaling and class weighting.")
