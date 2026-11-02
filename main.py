"""Synent Data Science Internship - Task 9: End-to-End Project: Diabetes Risk Screening App
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'pima': {'files': ['diabetes.csv', 'pima-indians-diabetes.csv'],
          'kaggle': 'uciml/pima-indians-diabetes-database',
          'name': 'Pima Indians Diabetes Database',
          'names': ['Pregnancies',
                    'Glucose',
                    'BloodPressure',
                    'SkinThickness',
                    'Insulin',
                    'BMI',
                    'DiabetesPedigreeFunction',
                    'Age',
                    'Outcome'],
          'no_header_file': 'pima-indians-diabetes.csv',
          'page': 'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database',
          'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task9_run(repo_dir):
    import joblib
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                                 confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)

    # 1) Data collection
    df = get_data("pima", repo_dir)
    df.columns = [str(c).strip() for c in df.columns]
    shape = [int(df.shape[0]), int(df.shape[1])]
    target = "Outcome"
    feats = [c for c in df.columns if c != target]

    # 2) Data cleaning: physiologically impossible zeros are really missing values
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    n_zero = {c: int((df[c] == 0).sum()) for c in zero_as_missing if c in df.columns}
    for c in n_zero:
        df[c] = df[c].replace(0, np.nan)
    df = df.drop_duplicates()
    out = _outdir(repo_dir)

    # 3) EDA
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    df[target].map({0: "No diabetes", 1: "Diabetes"}).value_counts().plot(kind="bar", ax=ax[0], color=["#2980b9", "#c0392b"])
    ax[0].set_title("Class balance")
    ax[0].tick_params(axis="x", rotation=0)
    (df[list(n_zero)].isna().mean() * 100).plot(kind="bar", ax=ax[1], color="#e67e22")
    ax[1].set_title("Missing values after treating zeros as missing (%)")
    fig.tight_layout()
    save_fig(fig, repo_dir, "eda_class_balance_missing.png")
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for a, f in zip(ax.ravel(), feats):
        sns.boxplot(data=df, x=target, y=f, ax=a)
        a.set_title(f)
    fig.tight_layout()
    save_fig(fig, repo_dir, "eda_boxplots_by_outcome.png")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    save_fig(fig, repo_dir, "eda_correlation_heatmap.png")

    # 4) Model building (imputation + scaling inside the pipeline, so there is no data leakage)
    X, y = df[feats], df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    def make(est):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", est)])
    candidates = {
        "Logistic Regression": make(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": make(RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, p in candidates.items():
        r = cross_validate(p, Xtr, ytr, cv=cv, scoring=["roc_auc", "recall", "f1"])
        rows.append({"model": name, "cv_roc_auc": round(r["test_roc_auc"].mean(), 3),
                     "cv_recall": round(r["test_recall"].mean(), 3), "cv_f1": round(r["test_f1"].mean(), 3)})
    cvdf = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    cvdf.to_csv(out / "cross_validation_results.csv", index=False)
    best_name = cvdf.iloc[0]["model"]
    best = candidates[best_name].fit(Xtr, ytr)
    proba = best.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {"accuracy": round(float(accuracy_score(yte, pred)), 3),
                    "precision": round(float(precision_score(yte, pred)), 3),
                    "recall": round(float(recall_score(yte, pred)), 3),
                    "f1": round(float(f1_score(yte, pred)), 3),
                    "roc_auc": round(float(roc_auc_score(yte, proba)), 3)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["No diabetes", "Diabetes"]).plot(ax=ax[0], colorbar=False, cmap="Blues")
    ax[0].set_title(f"Confusion matrix ({best_name})")
    RocCurveDisplay.from_predictions(yte, proba, ax=ax[1], name=best_name)
    ax[1].set_title("ROC curve")
    save_fig(fig, repo_dir, "model_evaluation.png")

    # 5) Deployment artefacts for the Streamlit app
    final = candidates[best_name].fit(X, y)
    mdir = Path(repo_dir) / "model"
    mdir.mkdir(exist_ok=True)
    joblib.dump(final, mdir / "model.joblib")
    info = {"features": feats, "target": target, "model_name": best_name, "test_metrics": test_metrics,
            "feature_stats": {f: {"min": float(X[f].min()), "max": float(X[f].max()), "median": float(X[f].median())} for f in feats}}
    (mdir / "feature_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    save_json(test_metrics, repo_dir, "test_metrics.json")

    pos = float(y.mean() * 100)
    if hasattr(final.named_steps["model"], "feature_importances_"):
        imp = pd.Series(final.named_steps["model"].feature_importances_, index=feats)
    else:
        imp = pd.Series(np.abs(final.named_steps["model"].coef_[0]), index=feats)
    top = imp.sort_values(ascending=False).index[:3].tolist()
    insights = [
        f"{pos:.0f}% of patients in the data have diabetes, so class weighting and recall were used to avoid missing positive cases.",
        f"Zeros in {', '.join(n_zero)} are biologically implausible placeholders for missing values, so they were treated as missing and imputed with medians inside the pipeline.",
        f"{best_name} gave the best cross-validated ROC-AUC and reached test recall {test_metrics['recall']} and ROC-AUC {test_metrics['roc_auc']}.",
        f"The most influential factors were {', '.join(top)}.",
        "This is an educational screening demo, not a medical diagnosis tool.",
    ]
    return {"headline": {"Best model": best_name, "Test ROC-AUC": test_metrics["roc_auc"], "Test recall": test_metrics["recall"],
                         "Test accuracy": test_metrics["accuracy"], "Patients": shape[0]},
            "insights": insights, "dataset_shape": shape,
            "files": ["model/model.joblib", "model/feature_info.json", "model_evaluation.png", "eda_correlation_heatmap.png",
                      "eda_boxplots_by_outcome.png", "cross_validation_results.csv", "test_metrics.json", "app.py"]}



if __name__ == "__main__":
    res = task9_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
