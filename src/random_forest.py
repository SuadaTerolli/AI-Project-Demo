import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from preprocessing import preprocess_data


def train_random_forest():
    X_train, X_test, y_train, y_test = preprocess_data()

    rf = RandomForestClassifier(
    n_estimators=500,          
    max_depth=16,              
    min_samples_split=10,     
    min_samples_leaf=5,       
    max_features="sqrt",
    class_weight="balanced_subsample", 
    random_state=42,
    n_jobs=-1
)

    rf.fit(X_train, y_train)

    y_proba=rf.predict_proba(X_test)[:, 1]


    custom_threshold = 0.3
    y_pred_tuned= (y_proba >= custom_threshold).astype(int)
    

    
    print("\n=== Random Forest Evaluation ===")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_tuned):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred_tuned):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_tuned))
    # Save model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n=== Top 10 Important Features ===")
    print(importance_df.head(10))

    importance_df.to_csv(
        os.path.join(MODELS_DIR, "feature_importance.csv"),
        index=False
    )


if __name__ == "__main__":
    train_random_forest()
