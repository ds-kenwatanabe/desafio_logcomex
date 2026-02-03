import joblib
import pandas as pd
from pathlib import Path
import warnings

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

try:
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

from src.preprocess import PreprocessConfig, build_preprocess_pipeline

PATH = "data/sample_data.parquet"


def print_class_balance(y: pd.Series, title: str):
    vc = y.value_counts()
    pct = y.value_counts(normalize=True) * 100
    out = pd.concat([vc, pct.rename("pct")], axis=1)
    print(f"\n=== Class balance: {title} ===")
    print(out)


def evaluate(y_true, y_pred, labels=None, title=""):
    print(f"\n================= {title} =================")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


def main():
    df = pd.read_parquet(PATH)
    df["registry_date"] = pd.to_datetime(df["registry_date"], errors="coerce")
    df = df.dropna(subset=["registry_date"]).sort_values("registry_date")

    cutoff = pd.Timestamp("2024-10-01")
    train = df[df["registry_date"] < cutoff].copy()
    valid = df[df["registry_date"] >= cutoff].copy()

    X_train = train.drop(columns=["channel"])
    y_train = train["channel"].astype(str)

    X_valid = valid.drop(columns=["channel"])
    y_valid = valid["channel"].astype(str)

    labels = sorted(y_train.unique())

    print_class_balance(y_train, "TRAIN")
    print_class_balance(y_valid, "VALID")

    # Preprocess
    cfg = PreprocessConfig(
        date_col="registry_date",
        target_col="channel",
        positive_label="VERMELHO",
    )
    pipe = build_preprocess_pipeline(cfg)

    Xt_train = pipe.fit_transform(X_train, y_train)
    Xt_valid = pipe.transform(X_valid)

    print("\nTrain matrix:", Xt_train.shape)
    print("Valid matrix:", Xt_valid.shape)

    # Model 1: Logistic Regression
    # Use solver SAGA for sparse/high-dimensional
    lr = LogisticRegression(
        solver="saga",
        max_iter=8000,
        tol=1e-3,
        class_weight="balanced",
        C=1.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr.fit(Xt_train, y_train)

    pred_lr = lr.predict(Xt_valid)
    evaluate(y_valid, pred_lr, labels=labels, title="LogisticRegression (saga, class_weight=balanced)")

    # Model 2: Linear SVM 
    svm = LinearSVC(class_weight="balanced")
    svm.fit(Xt_train, y_train)
    pred_svm = svm.predict(Xt_valid)
    evaluate(y_valid, pred_svm, labels=labels, title="LinearSVC (class_weight=balanced)")

    # Oversampling
    if IMBLEARN_AVAILABLE:
        print("\n--- Optional: RandomOverSampler on training set ---")
        ros = RandomOverSampler(random_state=42)
        Xt_train_os, y_train_os = ros.fit_resample(Xt_train, y_train)

        print_class_balance(pd.Series(y_train_os), "TRAIN after oversampling")

        lr_os = LogisticRegression(
            solver="saga",
            max_iter=8000,
            tol=1e-3,
            class_weight=None,
            C=1.0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            lr_os.fit(Xt_train_os, y_train_os)

        pred_lr_os = lr_os.predict(Xt_valid)
        evaluate(y_valid, pred_lr_os, labels=labels, title="LogReg (saga) + RandomOverSampler")
    else:
        print("\n[INFO] imbalanced-learn not installed; skipping oversampling. "
              "Install with: pip install imbalanced-learn")

    # Save artifacts 
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(pipe, "artifacts/preprocess.joblib")
    joblib.dump(lr, "artifacts/model_lr.joblib")
    joblib.dump(svm, "artifacts/model_svm.joblib")

    print("\nSaved artifacts to ./artifacts")


if __name__ == "__main__":
    main()
