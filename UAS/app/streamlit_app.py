"""
streamlit_app.py
Web deployment for the Credit Score classifier.

Run locally:
    streamlit run app/streamlit_app.py

Provides:
- A form for all model input features (raw-shaped, the pipeline cleans them).
- One-click "Load test case" buttons for Good / Poor / Standard so the grading
  screenshots cover every class.
- Prediction with confidence and full probability distribution.
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

# Make src importable for both local (`app/`) and deployed layouts.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from inference import InferencePipeline  # noqa: E402

st.set_page_config(page_title="Credit Score Classifier",
                   page_icon="!", layout="wide")

MODEL_DIR = os.path.join(ROOT, "models")
TEST_CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "test_cases.json")


@st.cache_resource
def load_pipeline():
    return InferencePipeline(model_dir=MODEL_DIR)


@st.cache_data
def load_test_cases():
    if os.path.exists(TEST_CASES_PATH):
        with open(TEST_CASES_PATH) as f:
            return json.load(f)
    return {}


pipe = load_pipeline()
test_cases = load_test_cases()

# ---- numeric vs categorical feature lists from training metadata ----------
RAW_COLS = pipe.metadata.get("raw_input_columns", [])
CAT_FIELDS = {
    "Occupation": ["Scientist", "Teacher", "Engineer", "Developer",
                   "Lawyer", "Doctor", "Journalist", "Manager",
                   "Accountant", "Architect", "Mechanic", "Writer",
                   "Entrepreneur", "Musician", "Media_Manager"],
    "Credit_Mix": ["Good", "Standard", "Bad"],
    "Payment_of_Min_Amount": ["Yes", "No"],
    "Payment_Behaviour": [
        "High_spent_Small_value_payments",
        "Low_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "Low_spent_Medium_value_payments",
        "High_spent_Large_value_payments",
        "Low_spent_Large_value_payments"],
}
# Everything raw that isn't categorical and isn't a free-text loan list.
NUM_FIELDS = [c for c in RAW_COLS
              if c not in CAT_FIELDS and c not in ("Type_of_Loan",)]

st.title("Credit Score Classifier")
st.caption(f"Model: **{pipe.metadata['best_model']}**  ·  "
           f"validation macro-F1 = **{pipe.metadata['macro_f1']:.3f}**  ·  "
           f"classes: {', '.join(pipe.classes)}")

# ---- test-case loader -----------------------------------------------------
st.subheader("Quick test cases")
st.write("Load a representative record for each class, then click Predict.")
cols = st.columns(len(test_cases) if test_cases else 1)
for i, (cls, record) in enumerate(test_cases.items()):
    if cols[i].button(f"Load '{cls}' case", use_container_width=True):
        st.session_state["form"] = record
        st.session_state["loaded_label"] = cls

form_values = st.session_state.get("form", {})
if "loaded_label" in st.session_state:
    st.info(f"Loaded a record whose true label is "
            f"**{st.session_state['loaded_label']}**.")

st.divider()
st.subheader("Customer features")

inputs = {}
c1, c2 = st.columns(2)
with c1:
    for field in NUM_FIELDS:
        default = form_values.get(field, 0.0)
        try:
            default = float(default) if default is not None else 0.0
        except (TypeError, ValueError):
            default = 0.0
        inputs[field] = st.number_input(field, value=default, format="%.4f")
with c2:
    for field, options in CAT_FIELDS.items():
        default = form_values.get(field)
        idx = options.index(default) if default in options else 0
        inputs[field] = st.selectbox(field, options, index=idx)
    # Type_of_Loan kept as a simple free-text box (pipeline reduces to a flag).
    inputs["Type_of_Loan"] = st.text_input(
        "Type_of_Loan (free text, optional)",
        value=str(form_values.get("Type_of_Loan", "") or ""))

st.divider()
if st.button("Predict credit score", type="primary",
             use_container_width=True):
    record = dict(inputs)
    if record["Type_of_Loan"].strip() == "":
        record["Type_of_Loan"] = None
    result = pipe.predict_with_confidence(record)[0]

    st.success(f"### Prediction: **{result['prediction']}**")
    st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    prob_df = pd.DataFrame({
        "Class": list(result["probabilities"].keys()),
        "Probability": list(result["probabilities"].values()),
    }).set_index("Class")
    st.bar_chart(prob_df)
    st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}),
                 use_container_width=True)

    if "loaded_label" in st.session_state:
        true = st.session_state["loaded_label"]
        if true == result["prediction"]:
            st.success(f"Matches the loaded true label ({true}).")
        else:
            st.warning(f"Loaded true label was {true}; "
                       f"model predicted {result['prediction']}.")
