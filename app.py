import os
import sys
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR
if os.path.basename(APP_DIR).lower() == "src":
    ROOT_DIR = os.path.dirname(APP_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rule_based import RuleBasedAttritionAgent
from src.preprocessing import preprocess_data

# ============================================================
# PATHS
# ============================================================
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PATH = os.path.join(ROOT_DIR, "data", "HR-Employee-Attrition.csv")

RF_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.csv")

RULE_FILE = os.path.join(ROOT_DIR, "src", "rule_based.py")
PREP_FILE = os.path.join(ROOT_DIR, "src", "preprocessing.py")


def file_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


CACHE_BUSTER = (
    file_mtime(RF_PATH),
    file_mtime(FEATURES_PATH),
    file_mtime(DATA_PATH),
    file_mtime(RULE_FILE),
    file_mtime(PREP_FILE),
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
<style>
/* ============================================================
   THEME TOKENS (default = light)
   ============================================================ */
:root {
  --bg: radial-gradient(1200px 600px at 10% 5%, #ffffff 0%, #f7f8fc 40%, #f2f4fb 100%);
  --text: #111827;

  --card-bg: rgba(255,255,255,0.80);
  --card-border: rgba(15, 23, 42, 0.08);

  --sidebar-bg: linear-gradient(180deg, #0b1220 0%, #0f1a33 60%, #0b1220 100%);
  --sidebar-text: #e5e7eb;

  --input-bg: #ffffff;
  --input-text: #111827;
  --input-border: rgba(15, 23, 42, 0.12);

  --dropdown-bg: #ffffff;
  --dropdown-text: #111827;

  --primary: #2563eb;
}

/* ============================================================
   DARK MODE overrides (based on PC theme)
   ============================================================ */
@media (prefers-color-scheme: dark) {
  :root {
    --bg: radial-gradient(1200px 600px at 10% 5%, #0b1220 0%, #0a0f1a 55%, #070b12 100%);
    --text: #e5e7eb;

    --card-bg: rgba(17, 24, 39, 0.72);
    --card-border: rgba(255,255,255,0.10);

    --sidebar-bg: linear-gradient(180deg, #050814 0%, #0b1220 60%, #050814 100%);
    --sidebar-text: #e5e7eb;

    --input-bg: #111827;
    --input-text: #e5e7eb;
    --input-border: rgba(255,255,255,0.14);

    --dropdown-bg: #111827;
    --dropdown-text: #e5e7eb;

    --primary: #60a5fa;
  }
}

/* ============================================================
   APP SURFACE
   ============================================================ */
.stApp {
  background: var(--bg);
  color: var(--text);
}
.block-container {
  max-width: 1400px;
  padding-top: 2.25rem;
  padding-bottom: 2.5rem;
}
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
p, li, span { color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }

/* Cards */
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  padding: 22px 22px;
  margin: 0 0 18px 0;
  backdrop-filter: blur(8px);
}
.kicker {
  font-size: 0.85rem;
  font-weight: 600;
  color: rgba(15, 23, 42, 0.70);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.4rem;
}
@media (prefers-color-scheme: dark) {
  .kicker { color: rgba(229, 231, 235, 0.72); }
}
.hero-title {
  font-size: 2.25rem;
  font-weight: 800;
  line-height: 1.12;
  margin: 0.25rem 0 0.5rem 0;
}
.hero-subtitle {
  font-size: 1.03rem;
  color: rgba(15, 23, 42, 0.72);
  max-width: 70ch;
  margin: 0.1rem 0 0 0;
}
@media (prefers-color-scheme: dark) {
  .hero-subtitle { color: rgba(229, 231, 235, 0.72); }
}

/* Buttons */
.stButton > button {
  background: linear-gradient(90deg, var(--primary) 0%, var(--primary) 50%, var(--primary) 100%);
  color: white;
  border: 0;
  border-radius: 12px;
  height: 3.1em;
  padding: 0 1.2rem;
  font-size: 0.98rem;
  font-weight: 600;
  box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
  transition: transform 0.08s ease-in-out, box-shadow 0.08s ease-in-out;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(37, 99, 235, 0.26);
}

/* ============================================================
   FIX INPUTS (number_input, selectbox, sliders)
   These are the ones that become unreadable in dark mode.
   ============================================================ */

/* Most Streamlit inputs render an <input> */
.stNumberInput input,
.stTextInput input {
  background: var(--input-bg) !important;
  color: var(--input-text) !important;
  border: 1px solid var(--input-border) !important;
}

/* Selectbox control */
div[data-baseweb="select"] > div {
  background: var(--input-bg) !important;
  color: var(--input-text) !important;
  border: 1px solid var(--input-border) !important;
}

/* Selectbox selected text */
div[data-baseweb="select"] span {
  color: var(--input-text) !important;
}

/* Dropdown menu list */
div[data-baseweb="popover"] div[role="listbox"] {
  background: var(--dropdown-bg) !important;
  color: var(--dropdown-text) !important;
  border: 1px solid var(--input-border) !important;
}

/* Dropdown menu items */
div[data-baseweb="popover"] div[role="option"] {
  background: var(--dropdown-bg) !important;
  color: var(--dropdown-text) !important;
}

/* Hover state */
div[data-baseweb="popover"] div[role="option"]:hover {
  filter: brightness(1.08);
}

/* Dataframe wrapper */
[data-testid="stDataFrame"] {
  border: 1px solid var(--card-border);
  border-radius: 14px;
  overflow: hidden;
}
/* STATUS boxes (force visible in both themes) */
.status {
  border-radius: 14px;
  padding: 16px 16px;
  border: 1px solid rgba(15, 23, 42, 0.10);
  margin-top: 12px;
}

/* Light mode */
.status.good {
  background: rgba(236, 253, 245, 0.85);
  border-left: 6px solid #10b981;
}
.status.bad {
  background: rgba(254, 242, 242, 0.85);
  border-left: 6px solid #ef4444;
}
.status .title { font-weight: 800; margin: 0 0 6px 0; }
.status .meta { margin: 0; }

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
  .status {
    border: 1px solid rgba(255,255,255,0.12);
  }
  .status.good {
    background: rgba(16, 185, 129, 0.12);  /* green tint */
    border-left: 6px solid #10b981;
  }
  .status.bad {
    background: rgba(239, 68, 68, 0.12);   /* red tint */
    border-left: 6px solid #ef4444;
  }
  .status .title,
  .status .meta {
    color: #e5e7eb !important;
  }
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL + FEATURE NAMES
# ============================================================
if not os.path.exists(RF_PATH):
    st.error(
        "Model file not found. Make sure this exists:\n"
        f"- {RF_PATH}\n\n"
        "Train and save the model before running the demo."
    )
    st.stop()

rf_model = joblib.load(RF_PATH)

if not os.path.exists(FEATURES_PATH):
    _ = preprocess_data()

feature_names = joblib.load(FEATURES_PATH)

# ============================================================
# CACHED DATA / SPLIT (keeps app consistent + faster)
# ============================================================
@st.cache_data(show_spinner=False)
def load_raw_df(cache_buster: tuple) -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_split(cache_buster: tuple):
    return preprocess_data()


@st.cache_data(show_spinner=False)
def get_rule_agent_train_calibrated(cache_buster: tuple) -> RuleBasedAttritionAgent | None:
    raw_df = load_raw_df(cache_buster)
    if raw_df.empty:
        return None
    X_train, X_test, y_train, y_test = get_split(cache_buster)
    raw_train_df = raw_df.loc[X_train.index].copy()
    return RuleBasedAttritionAgent(calibration_df=raw_train_df)

# ============================================================
# HELPERS
# ============================================================
def build_encoded_input(
    feature_names_list: list[str],
    age: int,
    monthly_income: int,
    distance: int,
    satisfaction: int,
    balance: int,
    years_company: int,
    overtime: str,
    total_years: int,
    years_manager: int,
    daily_rate: int,
    hourly_rate: int,
    monthly_rate: int,
) -> pd.DataFrame:
    x = pd.DataFrame(0, index=[0], columns=feature_names_list)

    values = [
        ("MonthlyIncome", monthly_income),
        ("Age", age),
        ("OverTime_Yes", 1 if overtime == "Yes" else 0),
        ("TotalWorkingYears", total_years),
        ("YearsAtCompany", years_company),
        ("DailyRate", daily_rate),
        ("YearsWithCurrManager", years_manager),
        ("MonthlyRate", monthly_rate),
        ("DistanceFromHome", distance),
        ("HourlyRate", hourly_rate),
        ("JobSatisfaction", satisfaction),
        ("WorkLifeBalance", balance),
    ]

    for col, val in values:
        if col in x.columns:
            x[col] = val

    return x


def build_raw_row(
    age: int,
    monthly_income: int,
    satisfaction: int,
    balance: int,
    years_company: int,
    overtime: str,
    distance: int,
    total_years: int,
    years_manager: int,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": monthly_income,
        "JobSatisfaction": satisfaction,
        "WorkLifeBalance": balance,
        "YearsAtCompany": years_company,
        "OverTime": overtime,
        "DistanceFromHome": distance,
        "TotalWorkingYears": total_years,
        "YearsWithCurrManager": years_manager,
    }])


def rf_predict_with_threshold(x_encoded: pd.DataFrame, threshold: float) -> tuple[int, float]:
    prob = float(rf_model.predict_proba(x_encoded)[0][1])
    pred = int(prob >= threshold)
    return pred, prob


def hybrid_sequential_predict(
    raw_row: pd.DataFrame,
    x_encoded: pd.DataFrame,
    threshold: float,
    rule_agent: RuleBasedAttritionAgent,
) -> tuple[int, float | None, str]:
    """
    Hybrid (rules-first, NO override):
    - If rule predicts 1 -> final 1
    - If rule predicts 0 -> final 0
    - If rule abstains -> RF decides (prob shown)
    Result explanation always reports BOTH rule and RF.
    """
    rule_pred = rule_agent.predict(raw_row).iloc[0] 

    rf_prob = float(rf_model.predict_proba(x_encoded)[0][1])
    rf_pred = int(rf_prob >= threshold)

    rule_text = "abstained" if pd.isna(rule_pred) else ("Yes" if int(rule_pred) == 1 else "No")
    rf_text = "Yes" if rf_pred == 1 else "No"

    if pd.notna(rule_pred):
        final_pred = int(rule_pred)
        explanation = (
            f"Rule-based prediction: {rule_text} (used as final decision). "
            f"Random Forest prediction: {rf_text} (probability {rf_prob:.2%}, threshold {threshold:.2f})."
        )
        return final_pred, None, explanation

    explanation = (
        f"Rule-based prediction: {rule_text}. "
        f"Random Forest prediction used as fallback: {rf_text} (probability {rf_prob:.2%}, threshold {threshold:.2f})."
    )
    return rf_pred, rf_prob, explanation


# ============================================================
# METRICS (rigorous + consistent with terminal)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_metrics(threshold: float, cache_buster: tuple) -> pd.DataFrame:
    raw_df = load_raw_df(cache_buster)
    if raw_df.empty:
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = get_split(cache_buster)

    raw_train_df = raw_df.loc[X_train.index].copy()
    raw_test_df = raw_df.loc[X_test.index].copy()
    rule_agent = RuleBasedAttritionAgent(calibration_df=raw_train_df)

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

    rf_proba_s = pd.Series(rf_model.predict_proba(X_test)[:, 1], index=y_test.index)
    rf_pred_tuned = (rf_proba_s >= threshold).astype(int)

    rf_auc = roc_auc_score(y_test, rf_proba_s)
    rf_precision = precision_score(y_test, rf_pred_tuned, zero_division=0)
    rf_recall = recall_score(y_test, rf_pred_tuned, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred_tuned, zero_division=0)

    final_preds = pd.Series(index=y_test.index, dtype=int)
    for idx in y_test.index:
        rp = rule_preds.loc[idx]
        rf_decision = int(rf_proba_s.loc[idx] >= threshold)

        if pd.isna(rp):
            final_preds.loc[idx] = rf_decision
        else:
            final_preds.loc[idx] = int(rp)

    hybrid_scores = rf_proba_s.copy()
    hybrid_scores.loc[rule_preds == 1] = 1.0

    hy_auc = roc_auc_score(y_test, hybrid_scores)
    hy_precision = precision_score(y_test, final_preds, zero_division=0)
    hy_recall = recall_score(y_test, final_preds, zero_division=0)
    hy_f1 = f1_score(y_test, final_preds, zero_division=0)

    return pd.DataFrame(
        {
            "Model": ["Rule-Based", "Random Forest (tuned)", "Hybrid Sequential (rules-first)"],
            "ROC-AUC": ["N/A", rf_auc, hy_auc],
            "Precision": [rb_precision, rf_precision, hy_precision],
            "Recall": [rb_recall, rf_recall, hy_recall],
            "F1 Score": [rb_f1, rf_f1, hy_f1],
            "Coverage": [coverage, 1.0, 1.0],
        }
    )

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
<div class="card">
  <div class="kicker">AI Systems Presentation</div>
  <div class="hero-title">Employee Attrition Prediction</div>
  <p class="hero-subtitle">
    This presentation compares three approaches: a rule-based system, a threshold-tuned Random Forest model,
    and a hybrid sequential system where rules decide first and machine learning completes coverage when rules abstain.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Navigation")
section = st.sidebar.radio(
    "Select a section",
    ["Overview", "Model Comparison", "Feature Importance", "Try a Prediction", "Conclusion"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Threshold tuning")
threshold = st.sidebar.slider(
    "Random Forest probability threshold (class = 1)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.01,
)

system_choice = st.sidebar.radio(
    "Prediction system",
    ["Random Forest", "Hybrid Sequential"],
    index=1,
)

# ============================================================
# OVERVIEW
# ============================================================
if section == "Overview":
    st.markdown(
        """
<div class="card">
  <div class="kicker">Section 1</div>
  <h2>Project Overview</h2>
  <p>
    Employee attrition is typically imbalanced (most employees stay). Accuracy alone can be misleading.
    This project reports ROC-AUC and uses threshold tuning to control the precision–recall trade-off.
  </p>

  <h3>Approach summary</h3>
  <ul>
    <li><b>Rule-Based</b>: explainable, but may abstain if no rule applies.</li>
    <li><b>Random Forest (tuned)</b>: outputs a probability score; threshold controls classification.</li>
    <li><b>Hybrid Sequential</b>: rules decide first; Random Forest fills undecided cases (no override).</li>
  </ul>

  <div class="callout">
    <p><b>Evaluation note:</b> Rules are calibrated on the training split and evaluated on the test split.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL COMPARISON
# ============================================================
elif section == "Model Comparison":
    st.markdown(
        """
<div class="card">
  <div class="kicker">Section 2</div>
  <h2>Model Performance Comparison</h2>
  <p>
    ROC-AUC measures ranking quality across thresholds. Precision, recall, and F1 are computed at the selected threshold.
    Coverage shows how often the rule-based system produces a decision.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    metrics_df = compute_metrics(threshold, CACHE_BUSTER)

    if metrics_df.empty:
        st.warning(f"Dataset not found at `{DATA_PATH}`. Put the CSV inside `data/` to compute metrics.")
    else:
        show_df = metrics_df.copy()
        for col in ["ROC-AUC", "Precision", "Recall", "F1 Score"]:
            show_df[col] = show_df[col].apply(lambda x: x if isinstance(x, str) else f"{x:.4f}")
        show_df["Coverage"] = show_df["Coverage"].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(show_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
elif section == "Feature Importance":
    st.markdown(
        """
<div class="card">
  <div class="kicker">Section 3</div>
  <h2>Feature Importance</h2>
  <p>
    The input form is aligned to the most influential features:
    MonthlyIncome, Age, OverTime, TotalWorkingYears, YearsAtCompany, DailyRate, YearsWithCurrManager,
    MonthlyRate, DistanceFromHome, and HourlyRate.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if os.path.exists(IMPORTANCE_PATH):
        importance_df = pd.read_csv(IMPORTANCE_PATH)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Top features (table)**")
            st.dataframe(importance_df.head(12), use_container_width=True)
        with right:
            st.markdown("**Top features (chart)**")
            st.bar_chart(importance_df.head(12).set_index("Feature")["Importance"])
    else:
        st.warning("Feature importance file not found. Generate `feature_importance.csv` in the models folder.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# INTERACTIVE PREDICTION
# ============================================================
elif section == "Try a Prediction":
    st.markdown(
        """
<div class="card">
  <div class="kicker">Section 4</div>
  <h2>Interactive Prediction</h2>
  <p>
    This form uses top Random Forest features. If the hybrid uses rules, probability may be unavailable
    because the decision is rule-based.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=6000, step=100)
        distance = st.slider("Distance From Home", 1, 30, 10)
        daily_rate = st.number_input("Daily Rate", min_value=1, max_value=2000, value=800, step=10)

    with c2:
        satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
        balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=2)
        years_company = st.slider("Years at Company", 0, 40, 5)
        monthly_rate = st.number_input("Monthly Rate", min_value=1, max_value=30000, value=15000, step=100)

    with c3:
        overtime = st.selectbox("OverTime", ["Yes", "No"], index=1)
        total_years = st.slider("Total Working Years", 0, 40, 10)
        years_manager = st.slider("Years with Manager", 0, 20, 3)
        hourly_rate = st.number_input("Hourly Rate", min_value=1, max_value=100, value=50, step=1)

    st.markdown("---")

    if st.button("Predict Attrition"):
        x_encoded = build_encoded_input(
            feature_names_list=feature_names,
            age=age,
            monthly_income=monthly_income,
            distance=distance,
            satisfaction=satisfaction,
            balance=balance,
            years_company=years_company,
            overtime=overtime,
            total_years=total_years,
            years_manager=years_manager,
            daily_rate=daily_rate,
            hourly_rate=hourly_rate,
            monthly_rate=monthly_rate,
        )

        raw_row = build_raw_row(
            age=age,
            monthly_income=monthly_income,
            satisfaction=satisfaction,
            balance=balance,
            years_company=years_company,
            overtime=overtime,
            distance=distance,
            total_years=total_years,
            years_manager=years_manager,
        )

        if system_choice == "Random Forest":
            pred, prob = rf_predict_with_threshold(x_encoded, threshold)
            explanation = f"Random Forest prediction computed with threshold {threshold:.2f}."
        else:
            rule_agent_live = get_rule_agent_train_calibrated(CACHE_BUSTER)
            if rule_agent_live is None:
                st.error("Dataset not found. Place HR-Employee-Attrition.csv inside data/ to calibrate rules.")
                st.stop()

            pred, prob, explanation = hybrid_sequential_predict(raw_row, x_encoded, threshold, rule_agent_live)

        prob_txt = (
            "Probability: not available for rule-based decisions."
            if prob is None
            else f"Estimated probability: {prob:.2%}"
        )

        if pred == 1:
            st.markdown(
                f"""
<div class="status bad">
  <p class="title">Higher attrition risk</p>
  <p class="meta">{prob_txt}</p>
  <p class="meta">{explanation}</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="status good">
  <p class="title">Lower attrition risk</p>
  <p class="meta">{prob_txt}</p>
  <p class="meta">{explanation}</p>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CONCLUSION
# ============================================================
elif section == "Conclusion":
    st.markdown(
        """
<div class="card">
  <div class="kicker">Section 5</div>
  <h2>Conclusion</h2>
  <ul>
    <li><b>Rule-based systems</b> provide strong interpretability, but may abstain and do not guarantee full coverage.</li>
    <li><b>Random Forest with threshold tuning</b> helps control the recall–precision trade-off for attrition.</li>
    <li><b>Hybrid sequential (rules-first)</b> uses rules when confident and falls back to ML when rules abstain.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
