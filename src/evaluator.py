import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score


class ModelEvaluator:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_prob = model.predict_proba(X_test)[:, 1]

    def metrics(self) -> dict:
        return {
            "accuracy" : round(accuracy_score(self.y_test, self.y_pred), 3),
            "precision": round(precision_score(self.y_test, self.y_pred), 3),
            "recall"   : round(recall_score(self.y_test, self.y_pred), 3),
            "f1_score" : round(f1_score(self.y_test, self.y_pred), 3),
            "roc_auc"  : round(roc_auc_score(self.y_test, self.y_prob), 3)
        }

    def print_report(self) -> None:
        m = self.metrics()
        print("accuracy :", m["accuracy"])
        print("precision:", m["precision"])
        print("recall   :", m["recall"])
        print("f1 score :", m["f1_score"])
        print("roc auc  :", m["roc_auc"])
        print()
        print(classification_report(self.y_test, self.y_pred, target_names=["ham", "spam"]))

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        return fpr, tpr

    def cross_validate(self, X, y, cv: int = 5) -> None:
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1")
        print("cv f1 scores:", scores.round(3))
        print("mean f1     :", round(scores.mean(), 3))
        print("std f1      :", round(scores.std(), 3))

    def error_analysis(self, X_test) -> pd.DataFrame:
        errors = X_test[self.y_pred != self.y_test].to_frame()
        errors["actual"] = self.y_test[self.y_pred != self.y_test].map({0: "ham", 1: "spam"})
        errors["predicted"] = self.y_pred[self.y_pred != self.y_test]
        errors["predicted"] = errors["predicted"].map({0: "ham", 1: "spam"})
        return errors