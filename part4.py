import math
import numpy as np
from common import (load_data, stratified_split, standardise, write_results,
                    sigmoid, train_logistic, confusion, rates,
                    neighbour_labels, write_csv, read_results)

X, y, value = load_data()
results = {}

tr, te = stratified_split(y)
ytr, yte = y[tr], y[te]
Xtr_o, Xte_o = X[tr], X[te]                      
Xtr_s, Xte_s = standardise(Xtr_o, Xte_o)         

# ---------------- Task 4.1: which split is purest? ----------------
def entropy(P, n):
    """-sum p log2 p over the two classes; 0 for a pure or empty node."""
    if n == 0:
        return 0.0
    h = 0.0
    for c in (P, n - P):
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h

def gini(P, n):
    if n == 0:
        return 0.0
    p = P / n
    return 1 - p ** 2 - (1 - p) ** 2

n_tr, P_tr = len(tr), int(ytr.sum())
root_h, root_g = entropy(P_tr, n_tr), gini(P_tr, n_tr)
results["root_entropy"] = root_h
results["root_gini"] = root_g

splits = {                                        
    "S1": lambda A: A[:, 5] == 1,
    "S2": lambda A: A[:, 6] == 1,
    "S3": lambda A: A[:, 2] > 0.5,
    "S4": lambda A: A[:, 0] <= 24,
}

split_rows = []
best_ig, best_ig_name, best_gg, best_gg_name = None, None, None, None
for name, cond in splits.items():
    yes = cond(Xtr_o)
    n_yes = int(yes.sum());  n_no = n_tr - n_yes
    p_yes = int(ytr[yes].sum());  p_no = P_tr - p_yes
    ig = root_h - (n_yes / n_tr * entropy(p_yes, n_yes) + n_no / n_tr * entropy(p_no, n_no))
    gg = root_g - (n_yes / n_tr * gini(p_yes, n_yes) + n_no / n_tr * gini(p_no, n_no))
    split_rows.append([name, n_yes, p_yes, n_no, p_no, ig, gg])
    if best_ig is None or ig > best_ig:           
        best_ig, best_ig_name = ig, name
    if best_gg is None or gg > best_gg:
        best_gg, best_gg_name = gg, name
write_csv("outputs/part4_splits.csv",
          "split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain", split_rows)
results["best_split_by_info_gain"] = best_ig_name
results["best_split_by_gini_gain"] = best_gg_name

# ---------------- Task 4.2: stump, voting ----------------
srow = next(r for r in split_rows if r[0] == best_ig_name)
rate_yes = srow[2] / srow[1]                      
rate_no = srow[4] / srow[3]
results["stump_rate_yes"] = rate_yes
results["stump_rate_no"] = rate_no
stump_prob = np.where(splits[best_ig_name](Xte_o), rate_yes, rate_no)

w, b, _ = train_logistic(Xtr_s, ytr, eta=0.1, n_updates=1000)   
p_log = sigmoid(Xte_s @ w + b)

K = int(read_results("outputs/part2_results.txt")["knn_best_k"])
lab = neighbour_labels(Xtr_s, ytr, Xte_s, K)
pos_nb = lab.sum(axis=1)
knn_share = pos_nb / K                            
pred_knn = (pos_nb >= (K + 1) // 2).astype(int)

pred_log = (p_log >= 0.3).astype(int)
pred_stump = (stump_prob >= 0.3).astype(int)
pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
pred_soft = (((p_log + knn_share + stump_prob) / 3) >= 0.3).astype(int)

vote_rows = []
for name, pred in [("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", pred_hard), ("soft_vote", pred_soft)]:
    tp, fp, tn, fn = confusion(yte, pred)
    r = rates(tp, fp, tn, fn)
    vote_rows.append([name, tp, fp, tn, fn, r["precision"], r["recall"], r["f1"]])
    print(name, "test cost (50*fp + 400*fn) =", 50 * fp + 400 * fn)   
write_csv("outputs/part4_voting.csv",
          "model,tp,fp,tn,fn,precision,recall,f1", vote_rows)

# ---------------- Task 4.3: two rounds of AdaBoost ----------------
yy = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
wt = np.full(8, 1 / 8)
ada_rows, errs, alphas = [], [], []
for rnd, h in ((1, h1), (2, h2)):
    eps = wt[h != yy].sum()                       
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = wt * np.exp(-alpha * yy * h)
    new = unnorm / unnorm.sum()                   
    for i in range(8):
        ada_rows.append([rnd, i + 1, wt[i], unnorm[i], new[i]])
    errs.append(eps)
    alphas.append(alpha)
    wt = new
write_csv("outputs/part4_adaboost.csv",
          "round,point,weight_before,weight_unnormalised,weight_after", ada_rows)
results.update({"round1_error": errs[0], "round1_alpha": alphas[0],
                "round2_error": errs[1], "round2_alpha": alphas[1]})
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
results["ensemble_accuracy"] = np.mean(ens == yy)

# ---------------- Task 4.4: out-of-bag share ----------------
results["oob_fraction"] = (1 - 1 / 560) ** 560

# ---------------- Task 4.5: parameter counts ----------------
def conv_out(size, f, pad, stride):
    return (size - f + 2 * pad) // stride + 1

mlp_h1 = 784 * 16 + 16                            
mlp_h2 = 16 * 16 + 16
mlp_out = 16 * 10 + 10

c1 = conv_out(28, 5, 2, 1);  cnn_conv1 = 5 * 5 * 1 * 6 + 6       
q1 = conv_out(c1, 2, 0, 2)
c2 = conv_out(q1, 5, 0, 1);  cnn_conv2 = 5 * 5 * 6 * 16 + 16
q2 = conv_out(c2, 2, 0, 2)
flat = q2 * q2 * 16
cnn_d1 = flat * 32 + 32
cnn_d2 = 32 * 10 + 10
print(f"shapes: conv1 {c1}x{c1}x6, pool1 {q1}x{q1}x6, conv2 {c2}x{c2}x16, "
      f"pool2 {q2}x{q2}x16, flatten {flat}")           # for README table R4.4

write_csv("outputs/part4_architecture.csv", "layer,parameters", [
    ["mlp_hidden1", mlp_h1], ["mlp_hidden2", mlp_h2], ["mlp_output", mlp_out],
    ["cnn_conv1", cnn_conv1], ["cnn_conv2", cnn_conv2],
    ["cnn_dense1", cnn_d1], ["cnn_dense2", cnn_d2]])
results["mlp_total"] = mlp_h1 + mlp_h2 + mlp_out
results["cnn_total"] = cnn_conv1 + cnn_conv2 + cnn_d1 + cnn_d2
results["fc_layer_params"] = (128 * 128 * 3) * 1024 + 1024
results["conv_layer_params"] = 3 * 3 * 3 * 64 + 64
results["rnn_total"] = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16   # Wxh+Whh+Why+bh+by

write_results("outputs/part4_results.txt", results)