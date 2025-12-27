import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6fa;
}
.main {
    background-color: #f4f6fa;
}
h1, h2, h3 {
    color: #1f2937;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.sidebar .sidebar-content {
    background-color: #5f2e70;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background-color: #1e40af;
}
.success-box {
    background-color: #ecfdf5;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #10b981;
}
.error-box {
    background-color: #fef2f2;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 Employee Attrition Prediction System")
st.markdown(
    "A comparison of **Rule-Based**, **Machine Learning**, and **Hybrid AI Systems** "
    "for predicting employee attrition."
)

# -----------------------------
# LOAD MODELS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Model Comparison",
        "Feature Importance",
        "Try a Prediction",
        "Conclusion"
    ]
)

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📌 Project Overview")

    st.markdown("""
    ### Models Used
    **Rule-Based System**
    - Expert-defined IF–THEN rules
    - Very accurate but limited coverage

    **Random Forest Classifier**
    - Data-driven machine learning model
    - Handles complex patterns

    **Hybrid Sequential System**
    - Rule-based decisions first
    - Random Forest handles undecided cases
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# MODEL COMPARISON
# -----------------------------
elif section == "Model Comparison":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Model Performance Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Rule-Based", "Random Forest", "Hybrid"],
        "Accuracy": [0.8932, 0.8435, 0.8469],
        "Precision": [0.7714, 0.5714, 0.5833],
        "Recall": [0.6585, 0.0851, 0.1489],
        "F1 Score": [0.7105, 0.1481, 0.2373],
        "Coverage": ["14%", "100%", "100%"]
    })

    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("""
    🔍 **Key Insight:**  
    Rule-based logic is precise but limited, while ML provides full coverage.
    The hybrid system balances both.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
elif section == "Feature Importance":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔍 Feature Importance (Random Forest)")

    importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")

    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)

        st.dataframe(importance_df.head(10), use_container_width=True)
        st.bar_chart(
            importance_df.head(10).set_index("Feature")["Importance"]
        )
    else:
        st.warning("Run random_forest.py to generate feature importance.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# INTERACTIVE PREDICTION
# -----------------------------
elif section == "Try a Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🧪 Try a Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        income = st.number_input("Monthly Income", 1000, 20000, 4000)
        distance = st.slider("Distance From Home", 1, 30, 10)

    with col2:
        satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        years_company = st.slider("Years at Company", 0, 40, 5)

    with col3:
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        total_years = st.slider("Total Working Years", 0, 40, 10)
        years_manager = st.slider("Years with Manager", 0, 20, 3)

    if st.button("🔮 Predict Attrition"):
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)

        input_df["Age"] = age
        input_df["MonthlyIncome"] = income
        input_df["DistanceFromHome"] = distance
        input_df["JobSatisfaction"] = satisfaction
        input_df["WorkLifeBalance"] = balance
        input_df["YearsAtCompany"] = years_company
        input_df["TotalWorkingYears"] = total_years
        input_df["YearsWithCurrManager"] = years_manager

        if overtime == "Yes" and "OverTime_Yes" in input_df.columns:
            input_df["OverTime_Yes"] = 1

        pred = rf_model.predict(input_df)[0]
        prob = rf_model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.markdown(
                f'<div class="error-box">⚠️ <b>High Attrition Risk</b><br>Probability: {prob:.2%}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="success-box">✅ <b>Low Attrition Risk</b><br>Probability: {prob:.2%}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# CONCLUSION
# -----------------------------
elif section == "Conclusion":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("✅ Conclusion")

    st.markdown("""
    - Rule-based systems provide **explainability**
    - Machine learning provides **scalability**
    - Hybrid AI combines both approaches effectively

    🎓 This project demonstrates **real-world AI system design**
    using symbolic and data-driven methods.
    """)

    st.success("Project ready for demo & grading!")
    st.markdown('</div>', unsafe_allow_html=True)