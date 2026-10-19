import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from models import step, sig

d = np.load("data.npz")
xtr, xte, ytr, yte = d["xtr"], d["xte"], d["ytr"], d["yte"]
s = np.load("scratch.npz")

sp = Perceptron(max_iter=1000, tol=1e-3, random_state=42).fit(xtr, ytr)
sl = LogisticRegression(random_state=42).fit(xtr, ytr)

ps = {
    "Perceptron (Scratch)": step(xte @ s["pw"] + s["pb"]),
    "Logistic (Scratch)": (sig(xte @ s["lw"] + s["lb"]) >= 0.5).astype(int),
    "Perceptron (sklearn)": sp.predict(xte),
    "Logistic (sklearn)": sl.predict(xte),
}

def mets(y, p):
    tp = np.sum((y == 1) & (p == 1))
    tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    acc = (tp + tn) / len(y)
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return np.array([[tn, fp], [fn, tp]]), acc, pre, rec, f1

p = ps["Logistic (Scratch)"]
cm, acc, pre, rec, f1 = mets(yte, p)
print("manual metrics - logistic (scratch)")
print(cm)
print("accuracy:", round(acc, 4), "| sklearn:", round(accuracy_score(yte, p), 4))
print("precision:", round(pre, 4), "| sklearn:", round(precision_score(yte, p, zero_division=0), 4))
print("recall:", round(rec, 4), "| sklearn:", round(recall_score(yte, p), 4))
print("f1:", round(f1, 4), "| sklearn:", round(f1_score(yte, p), 4))
print("match:", np.array_equal(cm, confusion_matrix(yte, p)) and np.isclose(f1, f1_score(yte, p)))

rows = []
for n, p in ps.items():
    cm = confusion_matrix(yte, p)
    print("\n" + n)
    print(cm)
    rows.append([n, cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
                 accuracy_score(yte, p), precision_score(yte, p, zero_division=0),
                 recall_score(yte, p), f1_score(yte, p)])

t = pd.DataFrame(rows, columns=["Model", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1"]).round(4)
print()
print(t.to_string(index=False))

b = t.sort_values(["Recall", "FN", "F1"], ascending=[False, True, False]).iloc[0]
print("\npreferred (highest recall, fewest FN):", b["Model"], "| recall:", b["Recall"], "| FN:", b["FN"])
