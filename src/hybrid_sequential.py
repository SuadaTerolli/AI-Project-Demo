import os
import pandas as pd
import joblib

from rule_based import RuleBasedAttritionAgent
from preprocessing import preprocess_data
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def run_sequential_system():
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = APP_DIR
    if os.path.basename(APP_DIR).lower() == "src":
        BASE_DIR = os.path.dirname(APP_DIR)

    DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    raw_df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess_data()

    raw_train_df = raw_df.loc[X_train.index].copy()
    raw_test_df = raw_df.loc[X_test.index].copy()

    rule_agent = RuleBasedAttritionAgent(calibration_df=raw_train_df)
    rule_preds = rule_agent.predict(raw_test_df)

    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

    rf_proba_all = pd.Series(rf_model.predict_proba(X_test)[:, 1], index=X_test.index)

    rf_threshold = 0.30
    rf_preds_all = (rf_proba_all >= rf_threshold).astype(int)

    final_preds = pd.Series(index=X_test.index, dtype=int)

    for idx in X_test.index:
        rp = rule_preds.loc[idx]
        if pd.isna(rp):
            final_preds.loc[idx] = int(rf_preds_all.loc[idx])
        else:
            final_preds.loc[idx] = int(rp)

    final_proba = rf_proba_all.copy()
    final_proba.loc[rule_preds == 1] = 1.0

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
