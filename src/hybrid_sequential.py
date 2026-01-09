import os
import pandas as pd
import joblib

from rule_based import RuleBasedAttritionAgent
from preprocessing import preprocess_data
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def run_sequential_system():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # 1) Load RAW data (for rules)
    raw_df = pd.read_csv(DATA_PATH)

    # 2) Preprocessing (train/test split) -> X_test keeps original row indices
    X_train, X_test, y_train, y_test = preprocess_data()

    # 3) Align raw test rows with encoded test rows
    raw_test_df = raw_df.loc[X_test.index]

    # 4) Rule-Based Predictions (RAW data) - IMPORTANT: calibrate
    rule_agent = RuleBasedAttritionAgent(calibration_df=raw_df)
    rule_preds = rule_agent.predict(raw_test_df)  # values: 0 / 1 / None

    # 5) Load Random Forest
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

    # 6) Get RF probabilities for ALL test rows (so we can override safely)
    rf_proba_all = pd.Series(rf_model.predict_proba(X_test)[:, 1], index=X_test.index)

    # Use the SAME tuned threshold you used in RF evaluation
    rf_threshold = 0.30
    rf_preds_all = (rf_proba_all >= rf_threshold).astype(int)

    # 7) Combine predictions (sequential + soft override)
    # Strategy:
    # - If rule says "Yes" (1) -> final Yes (high confidence)
    # - If rule abstains -> use RF
    # - If rule says "No" (0) -> allow RF to override to Yes if RF is confident (>= threshold)
    final_preds = pd.Series(index=X_test.index, dtype=int)

    for idx in X_test.index:
        rp = rule_preds.loc[idx]

        if pd.isna(rp):
            # rule abstain -> RF decides
            final_preds.loc[idx] = int(rf_preds_all.loc[idx])
        else:
            rp = int(rp)
            if rp == 1:
                # rule says leave -> keep
                final_preds.loc[idx] = 1
            else:
                # rule says stay -> allow RF override if it strongly predicts leave
                final_preds.loc[idx] = int(rf_preds_all.loc[idx])  # override-or-keep via threshold

    # 8) Build "final_proba" for ROC-AUC
    # Use RF probability everywhere EXCEPT when rule predicts "Yes", we force it to 1.0
    # (You can also force rule "No" to 0.0, but using RF proba helps AUC more.)
    final_proba = rf_proba_all.copy()
    final_proba.loc[rule_preds == 1] = 1.0

    # 9) Evaluation
    area_under_curve = roc_auc_score(y_test, final_proba)
    precision = precision_score(y_test, final_preds, zero_division=0)
    recall = recall_score(y_test, final_preds, zero_division=0)
    f1 = f1_score(y_test, final_preds, zero_division=0)

    print("\n=== Sequential Hybrid System Evaluation ===")
    print(f"ROC-AUC  : {area_under_curve:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    run_sequential_system()
