import numpy as np

def step(z):
    return (np.asarray(z) >= 0).astype(int)

def sig(z):
    return 1 / (1 + np.exp(-z))

def perc(x, y, lr=0.01, ep=100, seed=42):
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1])
    b = 0.0
    errs = []
    for _ in range(ep):
        e = 0
        for i in rng.permutation(len(y)):
            p = int(x[i] @ w + b >= 0)
            d = lr * (y[i] - p)
            w += d * x[i]
            b += d
            e += int(y[i] != p)
        errs.append(e)
        if e == 0:
            break
    return w, b, errs

def logr(x, y, lr=0.1, ep=1000):
    m = len(y)
    w = np.zeros(x.shape[1])
    b = 0.0
    loss = []
    for _ in range(ep):
        p = sig(x @ w + b)
        q = np.clip(p, 1e-12, 1 - 1e-12)
        loss.append(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        w -= lr * (x.T @ (p - y) / m)
        b -= lr * np.mean(p - y)
    return w, b, loss
