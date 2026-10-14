# Table 4: primary final-epoch comparisons of MVST+CVA (fold-level inference)
# Input : all_methods_long_v3.csv (one row per method, seed and fold)
# Output: difference in fold-level mean accuracy (pp), 95% hierarchical-bootstrap CI,
#         number of folds favoring MVST+CVA, paired t-test on the four fold means (df = 3),
#         and Holm-Bonferroni adjustment across the five comparisons.
import numpy as np
import pandas as pd
from scipy import stats

CSV = "all_methods_long_v3.csv"
METRIC = "final_epoch_accuracy"
REFERENCE = "mvst_cva"
COMPARISONS = [("mvst_concat", "MVST (concat)"), ("stgcn", "ST-GCN"),
               ("skateformer", "SkateFormer"), ("ctrgcn", "CTR-GCN"),
               ("sa_tdgformer", "SA-TDGFormer")]
N_BOOT, BOOT_SEED = 10000, 0


def hierarchical_bootstrap(a, b, folds, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample folds with replacement, then seeds within each resampled fold."""
    rng = np.random.default_rng(seed)
    va = {f: a[a.fold == f][METRIC].values for f in folds}
    vb = {f: b[b.fold == f][METRIC].values for f in folds}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        fd = []
        for f in rng.choice(folds, size=len(folds), replace=True):
            fd.append(rng.choice(va[f], size=len(va[f]), replace=True).mean()
                      - rng.choice(vb[f], size=len(vb[f]), replace=True).mean())
        diffs[i] = np.mean(fd)
    return np.percentile(diffs, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


# The CSV in the repository uses ";" as separator and "," as decimal mark
df = pd.read_csv(CSV, sep=None, engine="python", encoding="utf-8-sig")
for c in ["accuracy", "final_epoch_accuracy", "f1_score", "final_epoch_f1"]:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].astype(str).str.replace(",", ".").astype(float)
ref = df[df.method == REFERENCE]
folds = sorted(ref.fold.unique())
rows = []
for key, label in COMPARISONS:
    other = df[df.method == key]
    fm_ref = ref.groupby("fold")[METRIC].mean().sort_index()
    fm_oth = other.groupby("fold")[METRIC].mean().sort_index()
    d = fm_ref - fm_oth
    lo, hi = hierarchical_bootstrap(ref, other, folds)
    _, p = stats.ttest_rel(fm_ref.values, fm_oth.values)
    rows.append({"Comparison": f"vs. {label}", "Delta (pp)": d.mean() * 100,
                 "CI low (pp)": lo * 100, "CI high (pp)": hi * 100,
                 "Folds favoring MVST+CVA": f"{(d > 0).sum()}/{len(d)}", "p": p})

out = pd.DataFrame(rows)
out["Holm-adj p"] = holm(out["p"])
pd.set_option("display.width", 120)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
