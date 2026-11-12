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