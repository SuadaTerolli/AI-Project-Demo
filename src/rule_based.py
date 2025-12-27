import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

class RuleBasedAttritionAgent:
    def __init__(self):
        # VERY conservative rules
        # Safe (No Attrition) rules first
        self.rules = [
            self.rule_very_stable_employee,
            self.rule_high_income_no_overtime_long_tenure,

            # Very strong attrition signals
            self.rule_extreme_risk_low_income_overtime,
            self.rule_extreme_dissatisfaction_overtime,
        ]

    # =========================
    # SAFE / NO-ATTRITION RULES
    # =========================

    def rule_very_stable_employee(self, row):
        if (
            row["YearsAtCompany"] >= 12 and
            row["JobSatisfaction"] >= 3 and
            row["WorkLifeBalance"] >= 3 and
            row["OverTime"] == "No"
        ):
            return 0
        return None

    def rule_high_income_no_overtime_long_tenure(self, row):
        if (
            row["MonthlyIncome"] >= 9000 and
            row["YearsAtCompany"] >= 8 and
            row["OverTime"] == "No"
        ):
            return 0
        return None

    # =========================
    # EXTREME ATTRITION RULES
    # =========================

    def rule_extreme_risk_low_income_overtime(self, row):
        if (
            row["MonthlyIncome"] <= 2500 and
            row["OverTime"] == "Yes" and
            row["JobSatisfaction"] <= 2
        ):
            return 1
        return None

    def rule_extreme_dissatisfaction_overtime(self, row):
        if (
            row["JobSatisfaction"] == 1 and
            row["WorkLifeBalance"] == 1 and
            row["OverTime"] == "Yes"
        ):
            return 1
        return None

    # =========================
    # INFERENCE ENGINE
    # =========================

    def predict_single(self, row):
        for rule in self.rules:
            result = rule(row)
            if result is not None:
                return result
        return None  # abstain

    def predict(self, df):
        return df.apply(self.predict_single, axis=1)

def evaluate_rule_based_system(df):
    agent = RuleBasedAttritionAgent()

    y_true = df["Attrition"].map({"Yes": 1, "No": 0})
    y_pred = agent.predict(df)

    # Keep only rows where a rule fired
    decided_mask = y_pred.notna()

    y_true_decided = y_true[decided_mask]
    y_pred_decided = y_pred[decided_mask]

    coverage = decided_mask.mean()
    accuracy = accuracy_score(y_true_decided, y_pred_decided)
    precision = precision_score(y_true_decided, y_pred_decided)
    recall = recall_score(y_true_decided, y_pred_decided)
    f1 = f1_score(y_true_decided, y_pred_decided)

    return accuracy,precision,recall, f1, coverage


# MAIN (OPTIONAL)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")

    df = pd.read_csv(DATA_PATH)

    accuracy, precision, recall, f1, coverage = evaluate_rule_based_system(df)

    print("\n=== Rule-Based System Evaluation ===")
    print(f"Coverage : {coverage:.2%}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__": main()