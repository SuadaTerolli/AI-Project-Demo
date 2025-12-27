import pandas as pd
import joblib
import os

from rule_based import RuleBasedAttritionAgent
from preprocessing import preprocess_data
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score


def run_sequential_system():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # 1. Load RAW data
    raw_df = pd.read_csv(DATA_PATH)

    # 2. Preprocessing (train/test split)
    X_train, X_test, y_train, y_test = preprocess_data()

    # 3. Align raw test rows with encoded test rows
    raw_test_df = raw_df.loc[X_test.index]

    # 4. Rule-Based Predictions (RAW data)
    rule_agent = RuleBasedAttritionAgent()
    rule_preds = rule_agent.predict(raw_test_df)

    # 5. Identify undecided rows
    undecided_mask = rule_preds.isna()

    # 6. Load Random Forest
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

    # 7. RF predictions ONLY for undecided rows
    rf_preds = rf_model.predict(X_test.loc[undecided_mask])

    # 8. Combine predictions
    final_preds = rule_preds.copy()
    final_preds.loc[undecided_mask] = rf_preds

    # Ensure correct type
    final_preds = final_preds.astype(int)

    # 9. Evaluation
    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)

    print("\n=== Sequential Hybrid System Evaluation (TEST SET) ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    run_sequential_system()