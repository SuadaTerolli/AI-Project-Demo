import os
import sys
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                  
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.preprocessing import preprocess_data
except ModuleNotFoundError:
    from preprocessing import preprocess_data


class RuleBasedAttritionAgent:
    """
    Conservative rule-based attrition system with slightly improved coverage.

    - Calibrates thresholds from a calibration dataset (quantiles) instead of fixed numbers.
    - Adds a few high-confidence rules aligned with top RF features.
    - Abstains (returns None) when no rule fires.
    """

    def __init__(self, calibration_df: pd.DataFrame | None = None):
        self.q = None
        if calibration_df is not None:
            self._calibrate(calibration_df)

        self.rules = [
            self.rule_very_stable_employee,
            self.rule_high_income_no_overtime_long_tenure,

            self.rule_extreme_risk_low_income_overtime,
            self.rule_extreme_dissatisfaction_overtime,

            self.rule_young_low_experience_overtime,
            self.rule_manager_instability_overtime,
            self.rule_long_commute_overtime,
        ]

    # =========================
    # CALIBRATION (DATA-DRIVEN)
    # =========================
    def _calibrate(self, df: pd.DataFrame):
        self.q = {
            "income_p25": float(df["MonthlyIncome"].quantile(0.25)),
            "income_p75": float(df["MonthlyIncome"].quantile(0.75)),
            "tenure_p75": float(df["YearsAtCompany"].quantile(0.75)),
            "age_p30": float(df["Age"].quantile(0.30)),
            "twy_p25": float(df["TotalWorkingYears"].quantile(0.25)) if "TotalWorkingYears" in df.columns else None,
            "dist_p75": float(df["DistanceFromHome"].quantile(0.75)) if "DistanceFromHome" in df.columns else None,
        }

    def _thr(self, key: str, fallback):
        if self.q is None:
            return fallback
        v = self.q.get(key, None)
        return fallback if v is None else v

    # =========================
    # SAFE / NO-ATTRITION RULES
    # =========================
    def rule_very_stable_employee(self, row):
        if (
            row["YearsAtCompany"] >= self._thr("tenure_p75", 12) and
            row["JobSatisfaction"] >= 3 and
            row["WorkLifeBalance"] >= 3 and
            row["OverTime"] == "No"
        ):
            return 0
        return None

    def rule_high_income_no_overtime_long_tenure(self, row):
        if (
            row["MonthlyIncome"] >= self._thr("income_p75", 9000) and
            row["YearsAtCompany"] >= 6 and
            row["OverTime"] == "No"
        ):
            return 0
        return None

    # =========================
    # EXTREME ATTRITION RULES
    # =========================
    def rule_extreme_risk_low_income_overtime(self, row):
        if (
            row["MonthlyIncome"] <= self._thr("income_p25", 2500) and
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
    # ADDED RULES (CONSERVATIVE)
    # =========================
    def rule_young_low_experience_overtime(self, row):
        twy_p25 = self._thr("twy_p25", None)
        if twy_p25 is None:
            return None

        if (
            row["Age"] <= self._thr("age_p30", 30) and
            row["TotalWorkingYears"] <= twy_p25 and
            row["OverTime"] == "Yes"
        ):
            return 1
        return None

    def rule_manager_instability_overtime(self, row):
        if "YearsWithCurrManager" not in row:
            return None

        if (
            row["YearsAtCompany"] >= 3 and
            row["YearsWithCurrManager"] <= 1 and
            row["OverTime"] == "Yes"
        ):
            return 1
        return None

    def rule_long_commute_overtime(self, row):
        dist_p75 = self._thr("dist_p75", None)
        if dist_p75 is None:
            return None

        if (
            row["DistanceFromHome"] >= dist_p75 and
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
        return None

    def predict(self, df: pd.DataFrame):
        return df.apply(self.predict_single, axis=1)


def evaluate_rule_based_on_test(raw_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Rigorous evaluation:
    - Calibrate thresholds on TRAIN raw rows only
    - Evaluate on TEST raw rows only
    - Coverage computed on TEST only
    """
    X_train, X_test, y_train, y_test = preprocess_data()

    raw_train_df = raw_df.loc[X_train.index].copy()
    raw_test_df = raw_df.loc[X_test.index].copy()

    agent = RuleBasedAttritionAgent(calibration_df=raw_train_df)

    rule_preds = agent.predict(raw_test_df)
    decided_mask = rule_preds.notna()
    coverage = float(decided_mask.mean())

    if decided_mask.sum() == 0:
        return 0.0, 0.0, 0.0, coverage

    y_true_decided = y_test.loc[decided_mask]
    y_pred_decided = rule_preds.loc[decided_mask].astype(int)

    precision = precision_score(y_true_decided, y_pred_decided, zero_division=0)
    recall = recall_score(y_true_decided, y_pred_decided, zero_division=0)
    f1 = f1_score(y_true_decided, y_pred_decided, zero_division=0)

    return precision, recall, f1, coverage


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "HR-Employee-Attrition.csv")
    raw_df = pd.read_csv(data_path)

    precision, recall, f1, coverage = evaluate_rule_based_on_test(raw_df)

    print("\n=== Rule-Based System Evaluation (TRAIN-calibrated, TEST-evaluated) ===")
    print(f"Coverage : {coverage:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main()
