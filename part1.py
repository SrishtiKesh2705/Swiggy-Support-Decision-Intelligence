import math
import numpy as np
from common import load_data, write_results

X, y, value = load_data()
x4 = X[:, 3]                      
results = {}


n = len(value)
mean_value = value.mean()
sd_value = value.std(ddof=1)                  
se_value = sd_value / math.sqrt(n)
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value
results.update({"n": n, "mean_value": mean_value, "sd_value": sd_value,
                "se_value": se_value, "ci95_low": ci95_low, "ci95_high": ci95_high})


z_stat = (mean_value - 8.0) / se_value
reject_h0 = abs(z_stat) > 1.96
results.update({"z_stat": z_stat, "reject_h0": reject_h0})


E = 0.5
n_required = math.ceil((1.96 * sd_value / E) ** 2)
results["n_required"] = n_required



def fit_line(x, t):
    """slope = cov(x,t)/var(x) with divisor n-1 (the n-1 cancels, shown for clarity)."""
    m = len(x)
    xm, tm = x.mean(), t.mean()
    cov = np.sum((x - xm) * (t - tm)) / (m - 1)
    var = np.sum((x - xm) ** 2) / (m - 1)
    slope = cov / var
    intercept = tm - slope * xm
    fitted = intercept + slope * x
    resid = t - fitted
    tss = np.sum((t - tm) ** 2)
    ess = np.sum((fitted - tm) ** 2)
    rss = np.sum(resid ** 2)
    return {"slope": slope, "intercept": intercept, "fitted": fitted, "resid": resid,
            "tss": tss, "ess": ess, "rss": rss,
            "r2": 1 - rss / tss, "rmse": math.sqrt(rss / m), "mae": np.mean(np.abs(resid))}



raw = fit_line(x4, value)
results.update({
    "reg_slope": raw["slope"], "reg_intercept": raw["intercept"],
    "tss": raw["tss"], "ess": raw["ess"], "rss": raw["rss"],
    "tss_equals_ess_plus_rss": abs(raw["tss"] - raw["ess"] - raw["rss"]) < 1e-6 * raw["tss"],
    "r2": raw["r2"], "rmse": raw["rmse"], "mae": raw["mae"],
})


mean4, sd4 = x4.mean(), x4.std(ddof=1)
z = (x4 - mean4) / sd4
m_, c_ = 0.0, 0.0
eta = 0.1

def cost(m_, c_):
    e = value - (m_ * z + c_)
    return np.mean(e ** 2)

gd_costs = {0: cost(m_, c_)}
for step in range(1, 501):
    e = value - (m_ * z + c_)
    dm = -(2 / n) * np.sum(z * e)
    dc = -(2 / n) * np.sum(e)
    m_ = m_ - eta * dm
    c_ = c_ - eta * dc
    if step in (1, 10, 100, 500):
        gd_costs[step] = cost(m_, c_)

gd_slope = m_ / sd4
gd_intercept = c_ - m_ * mean4 / sd4
results.update({
    "gd_J0": gd_costs[0], "gd_J1": gd_costs[1], "gd_J10": gd_costs[10],
    "gd_J100": gd_costs[100], "gd_J500": gd_costs[500],
    "gd_slope": gd_slope, "gd_intercept": gd_intercept,
    "gd_slope_gap": abs(gd_slope - raw["slope"]),
})


def half_sds(fitted, resid):
    order = np.argsort(fitted, kind="stable")      
    lower, upper = order[:400], order[400:]
    return resid[lower].std(ddof=1), resid[upper].std(ddof=1)

lo_raw, up_raw = half_sds(raw["fitted"], raw["resid"])
results.update({"resid_sd_lower_raw": lo_raw, "resid_sd_upper_raw": up_raw,
                "resid_sd_ratio_raw": up_raw / lo_raw})

ln_x4, ln_val = np.log(x4), np.log(value)
ll = fit_line(ln_x4, ln_val)
lo_log, up_log = half_sds(ll["fitted"], ll["resid"])
results.update({
    "loglog_intercept": ll["intercept"], "loglog_slope": ll["slope"],
    "loglog_r2": ll["r2"], "loglog_rmse": ll["rmse"],
    "resid_sd_lower_log": lo_log, "resid_sd_upper_log": up_log,
    "resid_sd_ratio_log": up_log / lo_log,
})


A = np.column_stack([np.ones(n), ln_x4, X[:, 0], X[:, 1], X[:, 2],
                     X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(A.T @ A, A.T @ ln_val)      
fit_mr = A @ beta
rss_mr = np.sum((ln_val - fit_mr) ** 2)
tss_mr = np.sum((ln_val - ln_val.mean()) ** 2)
mr_r2 = 1 - rss_mr / tss_mr
p = 7
mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
for i in range(8):
    results[f"mr_beta_{i}"] = beta[i]
results["mr_r2"] = mr_r2
results["mr_adj_r2"] = mr_adj_r2
results["loglog_adj_r2"] = 1 - (1 - ll["r2"]) * (n - 1) / (n - 1 - 1)   # p = 1

write_results("outputs/part1_results.txt", results)