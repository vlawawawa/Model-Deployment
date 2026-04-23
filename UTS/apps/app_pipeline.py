"""
Run with: streamlit run apps/app_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.config import ARTIFACT_CLASSIFIER, ARTIFACT_REGRESSOR
from src.utils.io import load_artifact


@st.cache_resource
def load_models():
    clf = load_artifact(ARTIFACT_CLASSIFIER)
    reg = load_artifact(ARTIFACT_REGRESSOR)
    return clf, reg


def make_prediction(features: dict):
    clf, reg = load_models()
    df = pd.DataFrame([features])

    placed = bool(clf.predict(df)[0])
    placed_prob = float(clf.predict_proba(df)[0][1])
    salary = float(reg.predict(df)[0]) if placed else None
    return placed, placed_prob, salary


def build_features_from_ui(
    gender, extra, backlogs, ssc, hsc, deg, cgpa, attend,
    entrance, technical, soft, internship, projects, work_exp, certs
) -> dict:
    return {
        "gender":                     gender,
        "ssc_percentage":             int(ssc),
        "hsc_percentage":             int(hsc),
        "degree_percentage":          int(deg),
        "cgpa":                       float(cgpa),
        "entrance_exam_score":        int(entrance),
        "technical_skill_score":      int(technical),
        "soft_skill_score":           int(soft),
        "internship_count":           int(internship),
        "live_projects":              int(projects),
        "work_experience_months":     int(work_exp),
        "certifications":             int(certs),
        "attendance_percentage":      int(attend),
        "backlogs":                   int(backlogs),
        "extracurricular_activities": extra,
    }


def main():
    st.set_page_config(page_title="UTS MD", layout="wide")

    # Sidebar 
    with st.sidebar:
        st.header("**About**")
        st.markdown(
            "This app predicts whether a student will be **placed** "
            "and estimates their **salary package** using a trained "
            "Random Forest pipeline.\n\n"
            "**Models:**\n"
            "- Classifier -> placement_status\n"
            "- Regressor  -> salary_package_lpa"
        )
        st.divider()

    st.title("UTS MD - Valentino - Student Placement Predictor")
    st.markdown("Fill in the student profile below and click **Predict**.")

    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal")
            gender  = st.selectbox("Gender", ["Male", "Female"])
            extra   = st.selectbox("Extracurricular Activities", ["Yes", "No"])
            backlogs = st.number_input("Backlogs", 0, 20, 0)

        with col2:
            st.subheader("Academic Scores")
            ssc   = st.slider("SSC Percentage",    50, 95, 70)
            hsc   = st.slider("HSC Percentage",    50, 95, 72)
            deg   = st.slider("Degree Percentage", 50, 95, 72)
            cgpa  = st.slider("CGPA",              4.0, 10.0, 7.5, step=0.01)
            attend = st.slider("Attendance %",     50, 100, 85)

        with col3:
            st.subheader("Skills & Experience")
            entrance  = st.slider("Entrance Exam Score",    30, 100, 65)
            technical = st.slider("Technical Skill Score",  30, 100, 65)
            soft      = st.slider("Soft Skill Score",       30, 100, 65)
            internship = st.number_input("Internship Count",     0, 10, 1)
            projects   = st.number_input("Live Projects",        0, 10, 1)
            work_exp   = st.number_input("Work Experience (months)", 0, 60, 6)
            certs      = st.number_input("Certifications",       0, 20, 2)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    features = build_features_from_ui(
        gender, extra, backlogs, ssc, hsc, deg, cgpa, attend,
        entrance, technical, soft, internship, projects, work_exp, certs
    )

    # Results
    if submitted:
        placed, prob, salary = make_prediction(features)

        st.divider()
        st.subheader("Prediction Results")

        col_a, col_b = st.columns(2)
        with col_a:
            if placed:
                st.success(f"**PLACED**")
                st.balloons()
            else:
                st.error(f"**NOT PLACED**")
            st.metric("Placement Probability", f"{prob:.1%}")

        with col_b:
            if placed and salary is not None:
                st.success(f"Estimated Salary :")
                st.metric("Predicted Salary Package", f"{salary:.2f} LPA")
            else:
                st.info("Salary prediction is only available for placed students.")

        # Visual probability bar
        st.progress(prob)


if __name__ == "__main__":
    main()