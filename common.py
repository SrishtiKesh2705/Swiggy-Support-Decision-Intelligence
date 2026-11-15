import csv
import numpy as np

def load_data(path="data/records.csv"):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)                      
        rows = list(reader)
    X = np.array([[float(v) for v in r[1:8]] for r in rows])  
    y = np.array([int(r[8]) for r in rows])                   
    value = np.array([float(r[9]) for r in rows])             
    return X, y, value

def stratified_split(y):
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        n_train = (7 * len(idx)) // 10    
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))

def standardise(X_train, X_test):
    mean = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)      
    return (X_train - mean) / sd, (X_test - mean) / sd

def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return f"{float(v):.4f}"

def write_results(path, results):
    with open(path, "w") as f:
        for k, v in results.items():
            f.write(f"{k} = {fmt(v)}\n")

def sigmoid(z):
    z = np.clip(z, -500, 500)               
    return 1.0 / (1.0 + np.exp(-z))

def bce(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)        
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_logistic(Xs, y, eta, n_updates, record=()):
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Xs @ w + b))
    for step in range(1, n_updates + 1):
        p = sigmoid(Xs @ w + b)              
        grad_w = (Xs.T @ (p - y)) / n        
        grad_b = np.mean(p - y)              
        w = w - eta * grad_w
        b = b - eta * grad_b
        if step in record:
            losses[step] = bce(y, sigmoid(Xs @ w + b))
    return w, b, losses


def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn

def safe_div(a, b):
    return a / b if b != 0 else 0.0            # brief: zero denominator -> 0

def rates(tp, fp, tn, fn):
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)                # also TPR
    spec = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "specificity": spec, "fpr": fpr}

def neighbour_labels(ref_X, ref_y, query_X, kmax):
    out = np.zeros((len(query_X), kmax), dtype=int)
    for i, q in enumerate(query_X):
        d = np.sqrt(np.sum((ref_X - q) ** 2, axis=1))      # Euclidean distance
        order = np.argsort(d, kind="stable")[:kmax]
        out[i] = ref_y[order]
    return out

def knn_from_labels(labels, K):
    counts = labels[:, :K].sum(axis=1)
    return counts / K, (counts >= (K + 1) // 2).astype(int)

def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")

def read_results(path):
    out = {}
    with open(path) as f:
        for line in f:
            if " = " in line:
                k, v = line.strip().split(" = ")
                out[k] = v
    return out