import numpy as np
import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.joblib")

# =========================
# GLOBALS (loaded from disk)
# =========================
_state = {
    "model": None,
    "scaler": None,
    "feature_columns": None,
    "global_bmi_median": 28.1,
    "best_threshold": 0.35,
    "model_accuracy": None
}


def _save():
    joblib.dump(_state, MODEL_PATH)


def _load():
    global _state
    if os.path.exists(MODEL_PATH):
        _state = joblib.load(MODEL_PATH)
        return True
    return False


def train_model(df, force=False):
    """Train and save model to disk. If already trained, just load from disk."""
    global _state

    # If saved model exists and we're not forcing a retrain, just load it
    if not force and _load():
        print("[OK] Model loaded from disk.")
        return

    df = df.copy()

    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    _state["global_bmi_median"] = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(_state["global_bmi_median"])
    df = df[df['gender'] != 'Other']

    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df = pd.get_dummies(df, columns=categorical_cols)

    X = df.drop('stroke', axis=1)
    y = df['stroke']
    _state["feature_columns"] = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    _state["scaler"] = StandardScaler()
    X_train_s = _state["scaler"].fit_transform(X_train)
    X_test_s = _state["scaler"].transform(X_test)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1.0,
        batch_size=64,
        max_iter=300,
        random_state=42
    )
    _state["model"] = CalibratedClassifierCV(mlp, method='sigmoid', cv=3)
    _state["model"].fit(X_res, y_res)

    # Threshold optimization
    y_proba = _state["model"].predict_proba(X_test_s)[:, 1]
    thresholds = np.linspace(0.1, 0.7, 60)
    best_f1, best_t = 0, 0.35
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        report = classification_report(y_test, y_pred_t, output_dict=True, zero_division=0)
        f1 = report.get('1', {}).get('f1-score', 0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    _state["best_threshold"] = best_t

    final_report = classification_report(
        y_test, (y_proba >= best_t).astype(int), output_dict=True
    )
    _state["model_accuracy"] = {
        'accuracy': final_report['accuracy'],
        'stroke_precision': final_report['1']['precision'],
        'stroke_recall': final_report['1']['recall'],
        'stroke_f1': final_report['1']['f1-score'],
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold': best_t
    }

    _save()
    print(f"[OK] Model trained & saved. Accuracy={_state['model_accuracy']['accuracy']*100:.1f}%  Threshold={best_t:.2f}")


def predict_patient(data):
    global _state
    if _state["model"] is None:
        if not _load():
            raise Exception("[ERROR] Model not trained.")

    df_input = pd.DataFrame([data])
    df_input['bmi'] = pd.to_numeric(df_input['bmi'], errors='coerce')
    df_input['bmi'] = df_input['bmi'].fillna(_state["global_bmi_median"])

    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_input = pd.get_dummies(df_input, columns=categorical_cols)
    df_input = df_input.reindex(columns=_state["feature_columns"], fill_value=0)

    X_scaled = _state["scaler"].transform(df_input)
    prob = float(_state["model"].predict_proba(X_scaled)[0][1])
    pred = 1 if prob >= _state["best_threshold"] else 0

    return pred, prob, _state["best_threshold"]


def explain_case(data):
    reasons = []
    score = 0

    age = data.get('age', 0)
    if age > 70:
        reasons.append({"factor": "Elderly (Age > 70)", "severity": "high",
                        "detail": "Stroke risk significantly increases with age."})
        score += 3
    elif age > 55:
        reasons.append({"factor": "Age > 55", "severity": "moderate",
                        "detail": "Risk begins to rise significantly after 55."})
        score += 1

    if data.get('hypertension') == 1:
        reasons.append({"factor": "Hypertension", "severity": "critical",
                        "detail": "Directly stresses brain blood vessels."})
        score += 4

    if data.get('heart_disease') == 1:
        reasons.append({"factor": "Heart Disease", "severity": "high",
                        "detail": "Increases risk of clots traveling to the brain."})
        score += 3

    glucose = data.get('avg_glucose_level', 0)
    if glucose > 160:
        reasons.append({"factor": "High Glucose (>160 mg/dL)", "severity": "high",
                        "detail": "Chronic high glucose damages arteries."})
        score += 2
    elif glucose > 100:
        reasons.append({"factor": "Elevated Glucose (100-160)", "severity": "moderate",
                        "detail": "Pre-diabetic range increases cardiovascular risk."})
        score += 1

    if data.get('bmi', 0) > 30:
        reasons.append({"factor": "Obesity (BMI > 30)", "severity": "moderate",
                        "detail": "Obesity is linked to hypertension and diabetes."})
        score += 1

    if data.get('smoking_status') == 'smokes':
        reasons.append({"factor": "Active Smoker", "severity": "high",
                        "detail": "Smoking accelerates artery hardening."})
        score += 2
    elif data.get('smoking_status') == 'formerly smoked':
        reasons.append({"factor": "Former Smoker", "severity": "moderate",
                        "detail": "Risk remains elevated for years after quitting."})
        score += 1

    overall = "low"
    if score >= 8:
        overall = "critical"
    elif score >= 6:
        overall = "high"
    elif score >= 4:
        overall = "moderate"

    return reasons, overall, score


def get_model_stats():
    return _state.get("model_accuracy")