import numpy as np
from common import load_data, write_results, write_csv

X, y, value = load_data()
X5 = X[:, :5]                                  
n = len(X5)
results = {}

# ---------------- Task 3.1: K-means from scratch ----------------
Z = (X5 - X5.mean(axis=0)) / X5.std(axis=0, ddof=1)    

def kmeans(Z, K):
    step = len(Z) // K                                  
    cent = Z[[i * step for i in range(K)]].copy()       
    labels = None
    passes = 0
    while True:
        passes += 1
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)   
        new = np.argmin(d, axis=1)                      
        if labels is not None and np.array_equal(new, labels):
            break                                       
        labels = new
        for k in range(K):
            members = labels == k
            if members.any():                           
                cent[k] = Z[members].mean(axis=0)
    wcss = np.sum((Z - cent[labels]) ** 2)              
    return labels, cent, wcss, passes

wcss, fits = {}, {}
for K in (2, 3, 4, 5, 6):
    labels, cent, w, passes = kmeans(Z, K)
    wcss[K], fits[K] = w, labels
    results[f"wcss_k{K}"] = w
    print(f"K={K}: wcss={w:.4f}, passes={passes}")     

# ---------------- Task 3.2: elbow rule ----------------
elbow_k = None
for K in (3, 4, 5):
    if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
        elbow_k = K
        break
results["elbow_k"] = elbow_k                            

# ---------------- Task 3.3: profile the K = 3 segments ----------------
lab3 = fits[3]
x3_means = [X[lab3 == k, 2].mean() for k in range(3)]  
order = np.argsort(x3_means)                            
seg_of = {int(order[i]): "ABC"[i] for i in range(3)}
seg = np.array([seg_of[int(k)] for k in lab3])

seg_rows = []
for s in "ABC":
    m = seg == s
    results[f"size_segment_{s}"] = int(m.sum())
    seg_rows.append([s, int(m.sum())] + [X[m, j].mean() for j in range(5)]
                    + [y[m].mean(), value[m].mean()])
write_csv("outputs/part3_segments.csv",
          "segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value",
          seg_rows)

# ---------------- Task 3.4: hierarchical clustering on the supplied table ----------------
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])

def name(c):
    return "+".join(f"E{i + 1}" for i in sorted(c))

def agglomerate(D, link):
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dists = [D[i, j] for i in clusters[a] for j in clusters[b]]
                if link == "single":
                    h = min(dists)                      
                elif link == "complete":
                    h = max(dists)                     
                else:
                    h = sum(dists) / len(dists)         
                if best is None or h < best[0]:
                    best = (h, a, b)
        h, a, b = best
        ca, cb = clusters[a], clusters[b]
        if min(cb) < min(ca):                           
            ca, cb = cb, ca
        merges.append((ca, cb, h))
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [sorted(ca + cb)]
    return merges

hier_rows = []
for link in ("single", "complete", "average"):
    merges = agglomerate(D, link)
    for i, (ca, cb, h) in enumerate(merges, start=1):
        hier_rows.append([link, i, name(ca), name(cb), h])

    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = int(np.argmax(gaps))                            
    below = g + 1                                       
    results[f"{link}_clusters_at_cut"] = 6 - below

    groups = [frozenset([i]) for i in range(6)]         
    for ca, cb, _ in merges[:below]:
        a, b = frozenset(ca), frozenset(cb)
        groups = [c for c in groups if c not in (a, b)] + [a | b]
    print(link, "cut ->", sorted(name(c) for c in groups))

write_csv("outputs/part3_hierarchical.csv",
          "linkage,merge_no,cluster_a,cluster_b,height", hier_rows)

# ---------------- Task 3.5: PCA by the seven steps ----------------
mean_vec = Z.mean(axis=0)                               
Xc = Z - mean_vec                                       
S = (Xc.T @ Xc) / (n - 1)                               
vals, vecs = np.linalg.eigh(S)                          
idx = np.argsort(vals)[::-1]                            
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):                                      
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
W = vecs[:, :2]                                         
proj = Xc @ W                                           

total = vals.sum()
results["eigenvalue_sum"] = total
for j in range(5):
    results[f"eigenvalue_{j + 1}"] = vals[j]
ratios = vals / total
cum = np.cumsum(ratios)
for j in range(5):
    results[f"explained_ratio_{j + 1}"] = ratios[j]
for j in range(5):
    results[f"cumulative_ratio_{j + 1}"] = cum[j]
results["components_for_80pct"] = int(np.argmax(cum >= 0.8)) + 1
for j in range(5):
    results[f"pc1_loading_X{j + 1}"] = vecs[j, 0]
for j in range(5):
    results[f"pc2_loading_X{j + 1}"] = vecs[j, 1]
for r in range(3):                                      
    results[f"row{r + 1}_pc1"] = proj[r, 0]
    results[f"row{r + 1}_pc2"] = proj[r, 1]
for s in "ABC":
    results[f"mean_pc1_segment_{s}"] = proj[seg == s, 0].mean()

write_results("outputs/part3_results.txt", results)