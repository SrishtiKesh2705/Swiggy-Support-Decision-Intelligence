import numpy as np
from common import (load_data, stratified_split, standardise, write_results,
                    sigmoid, bce, train_logistic, confusion, rates,
                    neighbour_labels, knn_from_labels, write_csv)

X, y, value = load_data()
results = {}

# ---------------- Task 2.1: stratified split and standardise ----------------
tr, te = stratified_split(y)
ytr, yte = y[tr], y[te]
Xtr_s, Xte_s = standardise(X[tr], X[te])      

results.update({
    "train_rows": len(tr), "train_positives": int(ytr.sum()),
    "test_rows": len(te), "test_positives": int(yte.sum()),
})

# ---------------- Task 2.2: logistic regression, eta = 0.1, 1000 updates ----------------
w, b, losses = train_logistic(Xtr_s, ytr, eta=0.1, n_updates=1000,
                              record=(0, 1, 10, 100, 1000))
for k in (0, 1, 10, 100, 1000):
    results[f"loss_epoch_{k}"] = losses[k]
for j in range(7):
    results[f"weight_X{j + 1}"] = w[j]
results["bias"] = b
results["test_bce"] = bce(yte, sigmoid(Xte_s @ w + b))

# ---------------- Task 2.3: learning-rate effect (200 updates each) ----------------
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr_s, ytr, eta=eta, n_updates=200, record=(200,))
    results[f"loss_epoch_200_eta_{eta}"] = l[200]

# ---------------- Task 2.4: odds ratios ----------------
for j in range(7):
    results[f"odds_ratio_X{j + 1}"] = np.exp(w[j])

# ---------------- Task 2.5: test evaluation at tau = 0.5 and 0.3 ----------------
p_test = sigmoid(Xte_s @ w + b)
p_train = sigmoid(Xtr_s @ w + b)

for tau in (0.5, 0.3):
    pred = (p_test >= tau).astype(int)                 
    tp, fp, tn, fn = confusion(yte, pred)
    r = rates(tp, fp, tn, fn)
    results[f"tau_{tau}_tp"] = tp
    results[f"tau_{tau}_fp"] = fp
    results[f"tau_{tau}_tn"] = tn
    results[f"tau_{tau}_fn"] = fn
    for name in ("accuracy", "precision", "recall", "f1", "specificity"):
        results[f"tau_{tau}_{name}"] = r[name]

results["blind_accuracy"] = np.mean(yte == 0)          

# ---------------- Task 2.6: threshold sweep ----------------
rows = []
for k in range(1, 10):
    tau = k / 10
    tp, fp, tn, fn = confusion(yte, (p_test >= tau).astype(int))
    r = rates(tp, fp, tn, fn)
    rows.append([tau, tp, fp, tn, fn, r["recall"], r["fpr"]])
write_csv("outputs/part2_thresholds.csv", "tau,tp,fp,tn,fn,tpr,fpr", rows)

# ---------------- Task 2.7: KNN with 5-fold CV ----------------
KMAX = 21
odd_ks = list(range(1, 22, 2))                         
pos_in_train = np.arange(len(tr))
fold_of = pos_in_train % 5                             

fold_f1 = {K: [] for K in odd_ks}
for f in range(5):
    held = fold_of == f
    ref = ~held                                        
    labels = neighbour_labels(Xtr_s[ref], ytr[ref], Xtr_s[held], KMAX)
    for K in odd_ks:
        _, pred = knn_from_labels(labels, K)
        tp, fp, tn, fn = confusion(ytr[held], pred)
        fold_f1[K].append(rates(tp, fp, tn, fn)["f1"])

cv_rows = [[K, np.mean(fold_f1[K])] for K in odd_ks]
write_csv("outputs/part2_knn_cv.csv", "K,mean_cv_f1", cv_rows)

best_k = odd_ks[0]
best_f1 = cv_rows[0][1]
for K, score in cv_rows:
    if score > best_f1:                                
        best_k, best_f1 = K, score
results["knn_best_k"] = best_k

# refit: all 560 training rows as reference, evaluate on the test rows
test_labels = neighbour_labels(Xtr_s, ytr, Xte_s, KMAX)
_, knn_pred = knn_from_labels(test_labels, best_k)
tp, fp, tn, fn = confusion(yte, knn_pred)
r = rates(tp, fp, tn, fn)
results.update({"knn_tp": tp, "knn_fp": fp, "knn_tn": tn, "knn_fn": fn,
                "knn_accuracy": r["accuracy"], "knn_precision": r["precision"],
                "knn_recall": r["recall"], "knn_f1": r["f1"]})

# ---------------- Task 2.8: threshold chosen by money ----------------
def money_cost(fp, fn):
    return 50 * fp + 400 * fn

best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20
    tp, fp, tn, fn = confusion(ytr, (p_train >= tau).astype(int))   
    c = money_cost(fp, fn)
    if best_cost is None or c < best_cost:             
        best_tau, best_cost = tau, c
results["cost_tau"] = best_tau
results["cost_train_at_tau"] = best_cost

_, fp, _, fn = confusion(yte, (p_test >= best_tau).astype(int))
results["cost_test_at_tau"] = money_cost(fp, fn)
_, fp, _, fn = confusion(yte, (p_test >= 0.5).astype(int))
results["cost_test_at_0.5"] = money_cost(fp, fn)
results["cost_test_flag_nobody"] = 400 * int(yte.sum())            
write_results("outputs/part2_results.txt", results)