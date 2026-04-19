"""
Run AFTER starting the FastAPI server:
  uvicorn apps.B_fastapi:app --reload
Then run:
  streamlit run apps/app_streamlit.py
"""

import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000"


def call_api(endpoint: str, features: dict):
    try:
        response = requests.post(f"{FASTAPI_URL}{endpoint}", json=features, timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI server. Start it first:\n\n"
            "`uvicorn apps.B_fastapi:app --reload`"
        )
        return None


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
    st.set_page_config(page_title="Student Placement - API Client", layout="wide")

    # Sidebar
    with st.sidebar:
        st.header("**About**")
        st.markdown(
            "This Streamlit app acts as a **frontend client** that "
            "sends requests to the FastAPI backend.\n\n"
        )
        st.divider()
        mode = st.radio("Prediction Mode", [
            "Classification (Placement)",
            "Regression (Salary)",
            "Full Prediction (Both)",
        ])

    st.title("Student Placement API Client")
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
        st.divider()

        if "Classification" in mode:
            result = call_api("/predict/placement", features)
            if result:
                st.subheader("Classification Result")
                if result["placed"]:
                    st.success(result["message"])
                    st.balloons()
                else:
                    st.error(result["message"])
                col_a, col_b = st.columns(2)
                col_a.metric("Placed",      "Yes" if result["placed"] else "No")
                col_b.metric("Probability", f"{result['probability']:.1%}")
                st.progress(result["probability"])

        elif "Regression" in mode:
            result = call_api("/predict/salary", features)
            if result:
                st.subheader("Regression Result")
                st.info(result["message"])
                st.metric("Predicted Salary Package", f"{result['salary_lpa']:.2f} LPA")

        else:  # Full
            result = call_api("/predict/full", features)
            if result:
                st.subheader("Full Prediction Result")
                if result["placed"]:
                    st.success(result["summary"])
                    st.balloons()
                else:
                    st.error(result["summary"])
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Placed",      "Yes" if result["placed"] else "No")
                col_b.metric("Probability", f"{result['probability']:.1%}")
                col_c.metric("Salary",      f"{result['salary_lpa']:.2f} LPA" if result["placed"] else "N/A")
                st.progress(result["probability"])

        # Show raw API response
        with st.expander("Raw API Response"):
            st.json(result)


if __name__ == "__main__":
    main()
