import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data(for_tree_model=True):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    df = pd.read_csv(DATA_PATH)

    drop_cols = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df.drop(columns=drop_cols, inplace=True)

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_names.pkl"))

    return X_train, X_test, y_train, y_test



