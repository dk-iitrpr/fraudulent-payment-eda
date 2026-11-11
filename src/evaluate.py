# src/evaluate.py
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)

from config import OUT_DIR as OUT

def evaluate(model, X_train, y_train, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "baseline_accuracy": baseline.score(X_test, y_test),
    }

    for k, v in metrics.items():
        print(f"{k:>18}: {v:.3f}")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.legend()
    plt.savefig(OUT / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    coefs = model.named_steps["logisticregression"].coef_[0]
    order = np.argsort(coefs)
    
    plt.figure(figsize=(6, 4))
    plt.barh(np.array(feature_names)[order], np.exp(coefs[order]))
    plt.axvline(1, color="k", ls="--")
    plt.xlabel("Odds ratio (per 1 SD increase)")
    plt.title("Feature effects")
    plt.savefig(OUT / "odds_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()

    return metrics