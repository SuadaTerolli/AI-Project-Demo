import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


class RuleBasedAttritionAgent:
    """
    Conservative rule-based attrition system with *slightly* improved coverage.

    Changes vs your original:
    - Calibrates thresholds from the dataset (quantiles) instead of fixed numbers like 9000/2500/12.
    - Adds a few additional *high-confidence* rules aligned with top RF features:
      Age, TotalWorkingYears, YearsWithCurrManager, DistanceFromHome (always combined with OverTime).
    - Still abstains (returns None) when no rule fires -> hybrid is NOT dependent on rules.
    """

    def __init__(self, calibration_df: pd.DataFrame | None = None):
        self.q = None
        if calibration_df is not None:
            self._calibrate(calibration_df)

        # Safe (No Attrition) rules first, then strong attrition signals
        self.rules = [
            # SAFE / NO-ATTRITION
            self.rule_very_stable_employee,
            self.rule_high_income_no_overtime_long_tenure,

            # EXTREME ATTRITION
            self.rule_extreme_risk_low_income_overtime,
            self.rule_extreme_dissatisfaction_overtime,

            # ADDED (still conservative, aligned with RF top features)
            self.rule_young_low_experience_overtime,
            self.rule_manager_instability_overtime,
            self.rule_long_commute_overtime,
        ]

    # =========================
    # CALIBRATION (DATA-DRIVEN)
    # =========================
    def _calibrate(self, df: pd.DataFrame):
        # Quantiles make rules general (works on other datasets too)
        self.q = {
            # Income thresholds
            "income_p25": float(df["MonthlyIncome"].quantile(0.25)),
            "income_p75": float(df["MonthlyIncome"].quantile(0.75)),

            # Tenure threshold (top quartile)
            "tenure_p75": float(df["YearsAtCompany"].quantile(0.75)),

            # Younger employees (lower 30%)
            "age_p30": float(df["Age"].quantile(0.30)),

            # Low experience (lower 25%)
            "twy_p25": float(df["TotalWorkingYears"].quantile(0.25)) if "TotalWorkingYears" in df.columns else None,

            # Long commute (top quartile)
            "dist_p75": float(df["DistanceFromHome"].quantile(0.75)) if "DistanceFromHome" in df.columns else None,
        }

    # Helper to safely get calibrated values (falls back to your original constants)
    def _thr(self, key: str, fallback):
        if self.q is None:
            return fallback
        v = self.q.get(key, None)
        return fallback if v is None else v

    # =========================
    # SAFE / NO-ATTRITION RULES
    # =========================
    def rule_very_stable_employee(self, row):
        # Original: YearsAtCompany >= 12
        # Improved: YearsAtCompany in top quartile (still "very stable", but less extreme)
        if (
            row["YearsAtCompany"] >= self._thr("tenure_p75", 12) and
            row["JobSatisfaction"] >= 3 and
            row["WorkLifeBalance"] >= 3 and
            row["OverTime"] == "No"
        ):
            return 0
        return None

    def rule_high_income_no_overtime_long_tenure(self, row):
        # Original: MonthlyIncome >= 9000, YearsAtCompany >= 8
        # Improved: income top quartile; tenure moderately high
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
        # Original: MonthlyIncome <= 2500
        # Improved: income bottom quartile (keeps rule strong but not dataset-specific)
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
        """
        Conservative: Age low + TotalWorkingYears low + OverTime yes.
        Matches top RF drivers: Age, TotalWorkingYears, OverTime.
        """
        twy_p25 = self._thr("twy_p25", None)
        if twy_p25 is None:
            return None  # dataset/app may not provide this column in some contexts

        if (
            row["Age"] <= self._thr("age_p30", 30) and
            row["TotalWorkingYears"] <= twy_p25 and
            row["OverTime"] == "Yes"
        ):
            return 1
        return None

    def rule_manager_instability_overtime(self, row):
        """
        Conservative: short time with manager + not brand-new at company + overtime.
        Matches top RF: YearsWithCurrManager, YearsAtCompany, OverTime.
        """
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
        """
        Conservative: long commute (top quartile) + overtime yes.
        Matches top RF: DistanceFromHome, OverTime.
        """
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
        return None  # abstain

    def predict(self, df: pd.DataFrame):
        return df.apply(self.predict_single, axis=1)


def evaluate_rule_based_system(df: pd.DataFrame):
    agent = RuleBasedAttritionAgent(calibration_df=df)

    y_true = df["Attrition"].map({"Yes": 1, "No": 0})
    y_pred = agent.predict(df)

    decided_mask = y_pred.notna()
    coverage = float(decided_mask.mean())

    if decided_mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0

    y_true_decided = y_true[decided_mask]
    y_pred_decided = y_pred[decided_mask].astype(int)

    precision = precision_score(y_true_decided, y_pred_decided, zero_division=0)
    recall = recall_score(y_true_decided, y_pred_decided, zero_division=0)
    f1 = f1_score(y_true_decided, y_pred_decided, zero_division=0)

    return precision, recall, f1, coverage


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")

    df = pd.read_csv(DATA_PATH)

    precision, recall, f1, coverage = evaluate_rule_based_system(df)

    print("\n=== Rule-Based System Evaluation ===")
    print(f"Coverage : {coverage:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main()
