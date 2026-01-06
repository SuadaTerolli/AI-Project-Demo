import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.rule_based import RuleBasedAttritionAgent
from src.preprocessing import preprocess_data


# -----------------------------
# PATHS (robust for src/ layout)
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR

# If app.py is inside "src", models/ and data/ are typically one level up
if os.path.basename(APP_DIR).lower() == "src":
    ROOT_DIR = os.path.dirname(APP_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PATH = os.path.join(ROOT_DIR, "data", "HR-Employee-Attrition.csv")


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")


# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(
    """
<style>
/* App background */
.stApp { background-color: #24102b; }

/* Main area */
section[data-testid="stMain"] { background-color: #424141; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #21201f; }

/* Text */
h1, h2, h3, h4, h5, h6, p, label, li, title {
    color: #e5e7eb;
    font-family: 'JetBrains Mono', monospace;
}

/* Cards */
.card {
    background-color: #424141;
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 25px;
}

/* Buttons */
.stButton > button {
    background-color: #21201f;
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    border: none;
}
.stButton > button:hover { background-color: #383636; }

/* Info boxes */
.success-box {
    background-color: #064e3b;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #10b981;
    color: #ecfdf5;
}
.error-box {
    background-color: #7f1d1d;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #ef4444;
    color: #fef2f2;
}
.info-box {
    background-color: #1f2937;
    padding: 16px;
    border-radius: 12px;
    border-left: 6px solid #60a5fa;
    color: #e5e7eb;
}
.small {
    font-size: 0.9rem;
    opacity: 0.95;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    '<h1 style="font-family: JetBrains Mono, monospace;">🤖 Employee Attrition Prediction System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "A comparison of **Rule-Based**, **Random Forest (threshold-tuned)**, and a **Hybrid Sequential** system."
)

# -----------------------------
# LOAD MODELS / METADATA
# -----------------------------
if not os.path.exists(os.path.join(MODELS_DIR, "random_forest.pkl")):
    st.error(
        f"Missing model file: {os.path.join(MODELS_DIR, 'random_forest.pkl')}. "
        "Run random_forest.py first to train and save the model."
    )
    st.stop()

rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

# feature_names.pkl is saved by preprocess_data() during training
if not os.path.exists(os.path.join(MODELS_DIR, "feature_names.pkl")):
    # Fallback: create it by running preprocess once (still deterministic)
    _ = preprocess_data()
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

rule_agent = RuleBasedAttritionAgent()


# -----------------------------
# HELPERS
# -----------------------------
def build_encoded_input(
    feature_names_list: list[str],
    age: int,
    income: int,
    distance: int,
    satisfaction: int,
    balance: int,
    years_company: int,
    overtime: str,
    total_years: int,
    years_manager: int,
) -> pd.DataFrame:
    """Create one-row encoded dataframe aligned to training columns."""
    x = pd.DataFrame(0, index=[0], columns=feature_names_list)

    # Numeric features used in UI
    for col, val in [
        ("Age", age),
        ("MonthlyIncome", income),
        ("DistanceFromHome", distance),
        ("JobSatisfaction", satisfaction),
        ("WorkLifeBalance", balance),
        ("YearsAtCompany", years_company),
        ("TotalWorkingYears", total_years),
        ("YearsWithCurrManager", years_manager),
    ]:
        if col in x.columns:
            x[col] = val

    # Categorical (only what we collect in UI)
    if overtime == "Yes":
        # With get_dummies(drop_first=True), the training column is typically OverTime_Yes
        if "OverTime_Yes" in x.columns:
            x["OverTime_Yes"] = 1

    return x


def build_raw_row(
    age: int,
    income: int,
    satisfaction: int,
    balance: int,
    years_company: int,
    overtime: str,
) -> pd.DataFrame:
    """Raw row in original (pre-encoding) feature space for rule-based inference."""
    return pd.DataFrame(
        [
            {
                "Age": age,
                "MonthlyIncome": income,
                "JobSatisfaction": satisfaction,
                "WorkLifeBalance": balance,
                "YearsAtCompany": years_company,
                "OverTime": overtime,
            }
        ]
    )


def rf_predict_with_threshold(x_encoded: pd.DataFrame, threshold: float) -> tuple[int, float]:
    """Return (pred, prob_of_1)."""
    prob = float(rf_model.predict_proba(x_encoded)[0][1])
    pred = int(prob >= threshold)
    return pred, prob


def hybrid_sequential_predict(
    raw_row: pd.DataFrame, x_encoded: pd.DataFrame, threshold: float
) -> tuple[int, float | None, str]:
    """
    Sequential hybrid:
      1) Rule-based tries first.
      2) If rule abstains, use tuned RF with threshold.
    Returns: (pred, prob_of_1 or None, explanation)
    """
    rule_pred = rule_agent.predict(raw_row).iloc[0]
    if pd.notna(rule_pred):
        pred = int(rule_pred)
        # Probability isn't a true calibrated probability for rules; keep it as None
        explanation = "Decided by rule-based system (high-confidence rule fired)."
        return pred, None, explanation

    pred, prob = rf_predict_with_threshold(x_encoded, threshold)
    explanation = f"Rule-based abstained → decided by Random Forest with threshold={threshold:.2f}."
    return pred, prob, explanation


@st.cache_data(show_spinner=False)
def compute_metrics(threshold: float) -> pd.DataFrame:
    """
    Computes:
      - Rule-based metrics on decided subset + coverage
      - Random Forest metrics with threshold tuning
      - Hybrid sequential metrics with same tuned RF for undecided rows
    """
    if not os.path.exists(DATA_PATH):
        # Don't crash the app if the dataset isn't shipped with the UI
        return pd.DataFrame()

    raw_df = pd.read_csv(DATA_PATH)

    # Split (encoded)
    X_train, X_test, y_train, y_test = preprocess_data()

    # Align raw test rows to encoded test rows by index
    raw_test_df = raw_df.loc[X_test.index].copy()

    # ---- Rule-Based (only where decided)
    rule_preds = rule_agent.predict(raw_test_df)
    decided_mask = rule_preds.notna()
    coverage = float(decided_mask.mean())

    if decided_mask.any():
        y_true_decided = y_test.loc[decided_mask]
        y_pred_decided = rule_preds.loc[decided_mask].astype(int)
        rb_precision = precision_score(y_true_decided, y_pred_decided, zero_division=0)
        rb_recall = recall_score(y_true_decided, y_pred_decided, zero_division=0)
        rb_f1 = f1_score(y_true_decided, y_pred_decided, zero_division=0)
    else:
        rb_precision = rb_recall = rb_f1 = 0.0

    # ---- Random Forest tuned
    rf_proba_all = rf_model.predict_proba(X_test)[:, 1]
    rf_pred_tuned = (rf_proba_all >= threshold).astype(int)

    rf_auc = roc_auc_score(y_test, rf_proba_all)
    rf_precision = precision_score(y_test, rf_pred_tuned, zero_division=0)
    rf_recall = recall_score(y_test, rf_pred_tuned, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred_tuned, zero_division=0)

    # ---- Hybrid sequential
    final_preds = rule_preds.copy()
    # For undecided, apply tuned RF prediction
    undecided_mask = ~decided_mask
    if undecided_mask.any():
        final_preds.loc[undecided_mask] = rf_pred_tuned[undecided_mask]
    final_preds = final_preds.astype(int)

    # Probabilities for AUC:
    # - for decided: use 0/1 rule output as a score (not calibrated, but consistent with your evaluation script)
    # - for undecided: use RF probability
    hybrid_scores = pd.Series(index=y_test.index, dtype=float)
    hybrid_scores.loc[decided_mask] = final_preds.loc[decided_mask].astype(float)
    hybrid_scores.loc[undecided_mask] = rf_proba_all[undecided_mask]

    hy_auc = roc_auc_score(y_test, hybrid_scores)
    hy_precision = precision_score(y_test, final_preds, zero_division=0)
    hy_recall = recall_score(y_test, final_preds, zero_division=0)
    hy_f1 = f1_score(y_test, final_preds, zero_division=0)

    df = pd.DataFrame(
        {
            "Model": ["Rule-Based", "Random Forest (tuned)", "Hybrid Sequential (tuned RF)"],
            "ROC-AUC": ["N/A", rf_auc, hy_auc],
            "Precision": [rb_precision, rf_precision, hy_precision],
            "Recall": [rb_recall, rf_recall, hy_recall],
            "F1 Score": [rb_f1, rf_f1, hy_f1],
            "Coverage": [coverage, 1.0, 1.0],
        }
    )
    return df


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio(
    "Select Section",
    ["Overview", "Model Comparison", "Feature Importance", "Try a Prediction", "Conclusion"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Threshold tuning")
threshold = st.sidebar.slider(
    "Random Forest probability threshold (class=1)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.01,
)
system_choice = st.sidebar.radio("Prediction system", ["Random Forest", "Hybrid Sequential"])


# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>📌 Project Overview</h2>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="overview-text">
<b>Rule-Based System</b>
<ul>
  <li>Expert-defined IF–THEN rules</li>
  <li>Very interpretable, but it can abstain (limited coverage)</li>
</ul>

<b>Random Forest Classifier (threshold-tuned)</b>
<ul>
  <li>Data-driven ML model</li>
  <li>Uses probability threshold tuning to improve recall on unbalanced data</li>
</ul>

<b>Hybrid Sequential System</b>
<ul>
  <li>Rule-based decides first when very confident</li>
  <li>Random Forest handles undecided cases using the same tuned threshold</li>
</ul>

<div class="info-box small">
<b>Note:</b> Lowering the threshold generally increases <b>recall</b> but may reduce <b>precision</b>.
Hybrid can look "worse" than RF if the rules add false positives/negatives on top of RF decisions.
</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# MODEL COMPARISON
# -----------------------------
elif section == "Model Comparison":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>📊 Model Performance Comparison</h2>", unsafe_allow_html=True)

    metrics_df = compute_metrics(threshold)

    if metrics_df.empty:
        st.warning(
            "Could not compute metrics because the dataset file was not found at:\n"
            f"`{DATA_PATH}`\n\n"
            "If you want the comparison table to update automatically, keep the CSV inside `data/`."
        )
    else:
        # Format nicely for display
        show_df = metrics_df.copy()
        for col in ["ROC-AUC", "Precision", "Recall", "F1 Score"]:
            show_df[col] = show_df[col].apply(lambda x: x if isinstance(x, str) else f"{x:.4f}")
        show_df["Coverage"] = show_df["Coverage"].apply(lambda x: f"{x:.0%}")
        st.dataframe(show_df, use_container_width=True)

        st.markdown(
            f"""
<div class="info-box small">
Current threshold: <b>{threshold:.2f}</b>.  
Try lowering it (e.g., 0.25 → 0.20) if you want higher recall, but expect more false positives.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
elif section == "Feature Importance":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>🔍 Feature Importance (Random Forest)</h2>", unsafe_allow_html=True)

    importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        st.dataframe(importance_df.head(12), use_container_width=True)
        st.bar_chart(importance_df.head(12).set_index("Feature")["Importance"])
    else:
        st.warning("Run random_forest.py to generate feature_importance.csv.")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# INTERACTIVE PREDICTION
# -----------------------------
elif section == "Try a Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>🧪 Try a Prediction</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        income = st.number_input("Monthly Income", 1000, 20000, 4000)
        distance = st.slider("Distance From Home", 1, 30, 10)

    with col2:
        satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
        balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
        years_company = st.slider("Years at Company", 0, 40, 5)

    with col3:
        overtime = st.selectbox("OverTime", ["Yes", "No"], index=1)
        total_years = st.slider("Total Working Years", 0, 40, 10)
        years_manager = st.slider("Years with Manager", 0, 20, 3)

    if st.button("Predict Attrition"):
        # Encoded input for RF
        x_encoded = build_encoded_input(
            feature_names_list=feature_names,
            age=age,
            income=income,
            distance=distance,
            satisfaction=satisfaction,
            balance=balance,
            years_company=years_company,
            overtime=overtime,
            total_years=total_years,
            years_manager=years_manager,
        )

        # Raw row for rule-based
        raw_row = build_raw_row(
            age=age,
            income=income,
            satisfaction=satisfaction,
            balance=balance,
            years_company=years_company,
            overtime=overtime,
        )

        if system_choice == "Random Forest":
            pred, prob = rf_predict_with_threshold(x_encoded, threshold)
            explanation = f"Random Forest probability threshold = {threshold:.2f}."
        else:
            pred, prob, explanation = hybrid_sequential_predict(raw_row, x_encoded, threshold)

        if pred == 1:
            prob_txt = "Probability: N/A (rule-based)" if prob is None else f"Probability: {prob:.2%}"
            st.markdown(
                f'<div class="error-box">⚠️ <b>High Attrition Risk</b><br>{prob_txt}'
                f'<br><span class="small">{explanation}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            prob_txt = "Probability: N/A (rule-based)" if prob is None else f"Probability: {prob:.2%}"
            st.markdown(
                f'<div class="success-box">✅ <b>Low Attrition Risk</b><br>{prob_txt}'
                f'<br><span class="small">{explanation}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# CONCLUSION
# -----------------------------
elif section == "Conclusion":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Conclusion</h2>", unsafe_allow_html=True)

    st.markdown(
        """
- **Rule-based systems** give strong explainability, but they may abstain.
- **Threshold-tuned Random Forest** helps handle unbalanced data by trading precision for recall.
- **Hybrid Sequential** can be best when rules are *high precision* and cover meaningful cases, while RF handles everything else.

If your hybrid recall/precision is lower than RF even with the same tuned RF:
- that usually means the **rules are making extra mistakes** on top of the RF decisions, not that RF needs an even lower threshold.
- improve hybrid by making rules **more conservative** (higher precision), or by letting rules *only* decide “safe negatives” (or only “extreme positives”) depending on your goal.
""",
    )

    st.markdown("</div>", unsafe_allow_html=True)
