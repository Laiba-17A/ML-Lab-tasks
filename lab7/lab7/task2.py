import numpy as np
import matplotlib.pyplot as plt
from models import step, sig, perc, logr

d = np.load("data.npz")
xtr, xte, ytr, yte = d["xtr"], d["xte"], d["ytr"], d["yte"]

pw, pb, errs = perc(xtr, ytr)
print("perceptron w:", pw.round(4), "b:", round(pb, 4))
print("perceptron first 5 preds:", step(xte[:5] @ pw + pb).tolist())

lw, lb, loss = logr(xtr, ytr)
pr = sig(xte @ lw + lb)
print("logistic w:", lw.round(4), "b:", round(lb, 4))
print("logistic first 5 preds:", (pr[:5] >= 0.5).astype(int).tolist())
print("logistic first 5 probs:", pr[:5].round(4).tolist())
print("final loss:", round(loss[-1], 4))

plt.plot(loss)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Logistic Regression (Scratch) - Loss vs Epochs")
plt.grid(True)
plt.savefig("loss.png", dpi=120)
plt.show()

np.savez("scratch.npz", pw=pw, pb=pb, lw=lw, lb=lb)
