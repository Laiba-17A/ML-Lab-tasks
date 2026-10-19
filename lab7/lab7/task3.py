import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, LogisticRegression
from models import step, sig

d = np.load("data.npz")
xtr, xte, ytr, yte = d["xtr"], d["xte"], d["ytr"], d["yte"]
s = np.load("scratch.npz")

sp = Perceptron(max_iter=1000, tol=1e-3, random_state=42).fit(xtr, ytr)
sl = LogisticRegression(random_state=42).fit(xtr, ytr)

print("sklearn perceptron w:", sp.coef_[0].round(4), "b:", sp.intercept_.round(4))
print("sklearn logistic w:", sl.coef_[0].round(4), "b:", sl.intercept_.round(4))
print("scratch perceptron w:", s["pw"].round(4), "b:", round(float(s["pb"]), 4))
print("scratch logistic w:", s["lw"].round(4), "b:", round(float(s["lb"]), 4))

ms = {
    "Perceptron (Scratch)": lambda g: step(g @ s["pw"] + s["pb"]),
    "Logistic (Scratch)": lambda g: (sig(g @ s["lw"] + s["lb"]) >= 0.5).astype(int),
    "Perceptron (sklearn)": sp.predict,
    "Logistic (sklearn)": sl.predict,
}

xa = np.vstack([xtr, xte])
a, c = np.meshgrid(np.linspace(xa[:, 0].min() - 0.5, xa[:, 0].max() + 0.5, 300),
                   np.linspace(xa[:, 1].min() - 0.5, xa[:, 1].max() + 0.5, 300))
g = np.c_[a.ravel(), c.ravel()]

fig, ax = plt.subplots(2, 2, figsize=(11, 9))
for k, (n, f) in zip(ax.ravel(), ms.items()):
    k.contourf(a, c, f(g).reshape(a.shape), alpha=0.25, cmap="coolwarm")
    k.scatter(xte[:, 0], xte[:, 1], c=yte, cmap="coolwarm", edgecolor="k", s=25)
    k.set_title(n)
    k.set_xlabel("Glucose (scaled)")
    k.set_ylabel("BMI (scaled)")
plt.tight_layout()
plt.savefig("boundaries.png", dpi=120)
plt.show()
