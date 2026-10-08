"""
SHAPxAdopt-GH -- revision experiments
=====================================
Runs every analysis requested in supervisory
Reports 1-5 on the cleaned survey workbook and writes all tables/figures to
./revision_outputs/.

WHAT THE SCRIPT DOES, IN ORDER (the S-codes match the thesis tables)
  STEP 0  Load the 300 responses, build the Low/Medium/High target from C5,
          one-hot encode the 16 categorical questions  -> 300 rows x 70 features
  STEP 1  Reproduce the original 70/15/15 split (seed 42) and the four original
          models, add LightGBM, bootstrap CIs, McNemar + effect sizes   (S3)
  STEP 2  Stratified 5-fold cross-validation for all five models     (S1)
  STEP 3  Repeat the whole pipeline with 10 different random seeds   (S2)
  STEP 4  Ablations: no institutional covariates, rare categories collapsed,
          alternative C5 thresholds, linear vs interaction LR   (S4, S7, S8, S9)
  STEP 5  Leave-one-institution-tier-out validation                 (S5)
  STEP 6  Measurement-noise perturbation test                       (S6)
  STEP 7  Per-tier (subgroup) audit                                  (S10)
  STEP 8  Model complexity, clean SHAP figures, target-proximity screen
                                                              (S11, S12, S13)
  STEP 9  Duplicate-profile check and duplicate-aware validation     (S14)
  STEP 10 Equalised-odds audit on out-of-fold predictions (all 300)  (S15)
  STEP 11 SHAP vs XGBoost gain importance; waterfalls per tier       (S16, S17)

Run:  python shapxadopt_revision_experiments.py --data AI_Readiness_Cleaned_Workbook_300_1.xlsx
"""
import argparse, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, label_binarize, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             accuracy_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import spearmanr, chi2_contingency
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ============================ CONFIGURATION ================================
SHEET = "Modeling_Data"           # sheet with reverse-coded items + composites
N_ROWS = 300                      # the sheet has notes under row 300
CONSTRUCTS = ["F1_score", "F2_score", "F3_reduced_score", "F4_score",
              "F5_score", "F6_score", "F7_score"]                   # 7 UTAUT composites
USAGE = ["Uses_LMS", "Uses_SMIS", "Uses_AI_Teaching", "Uses_AI_Admin",
         "Uses_VideoConf", "Uses_DigitalLibrary"]                   # 6 binary usage flags
INSTITUTIONAL = ["A1", "A2", "A5", "A6", "A7", "A8"]                # role, type, enrolment,
                                                                    # staff, ICT unit, years in role
SURVEY_CATEGORICAL = ["C3", "C4", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
CATEGORICAL = INSTITUTIONAL + SURVEY_CATEGORICAL                    # 16 one-hot encoded
TIER_COL = "A2"                                                     # institution type
TARGET_SOURCE = "C5"                                                # 0-10 readiness rating
THRESHOLDS_MAIN = (5, 7)          # Low <= 5, Medium 6-7, High >= 8
THRESHOLDS_ALT = (4, 8)           # Low <= 4, Medium 5-8, High >= 9
SEEDS = [42, 7, 13, 21, 99, 123, 2024, 88, 55, 3]
NOISE_SD, NOISE_TRIALS, N_BOOT, RARE = 0.2, 20, 5000, 10
CLASS_NAMES = ["Low", "Medium", "High"]
MODEL_ORDER = ["XGBoost (proposed)", "Logistic Regression", "Random Forest", "SVM (RBF)", "LightGBM"]
OUT = "revision_outputs"
# Original XGBoost settings (thesis Section 2.9)
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=2.0, eval_metric="mlogloss",
                  early_stopping_rounds=20)
# ==========================================================================


# --------------------------- STEP 0: DATA ---------------------------------
def make_target(c5, thr):
    """Convert the 0-10 readiness rating into 0 = Low, 1 = Medium, 2 = High."""
    lo, mid = thr
    return np.where(c5 <= lo, 0, np.where(c5 <= mid, 1, 2)).astype(int)


def load(path):
    df = pd.read_excel(path, sheet_name=SHEET).iloc[:N_ROWS].reset_index(drop=True)
    return df


def encode(df, collapse_rare=False, drop=()):
    """One-hot encode the categorical questions (each answer option -> 0/1 column).
    collapse_rare: merge answer options chosen by < RARE respondents into 'Other'."""
    cats = [c for c in CATEGORICAL if c not in drop]
    X = df[CONSTRUCTS + USAGE + cats].copy()
    if collapse_rare:
        for c in cats:
            vc = X[c].value_counts()
            X[c] = X[c].where(~X[c].isin(vc[vc < RARE].index), "Other")
    X = pd.get_dummies(X, columns=cats, dtype=float)
    X.columns = [str(c) for c in X.columns]
    return X


# --------------------------- HELPERS --------------------------------------
def models(seed):
    """The proposed model and four baselines, configured as in the thesis.
    'xgb' is a marker: XGBoost is fitted separately because it needs a
    validation set for early stopping."""
    return {
        "XGBoost (proposed)": "xgb",
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=seed),   # L2 is the default
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=seed),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)),
        # LightGBM rejects some characters in the survey answer text, so it is given plain arrays
        "LightGBM": make_pipeline(FunctionTransformer(np.asarray),
                                  lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                                     subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)),
    }


def smote(Xtr, ytr, seed):
    """SMOTE on training data only. If a class is too small for 5 neighbours
    (happens only in leave-one-tier-out), k is reduced; a class with a single
    case cannot be oversampled and is left as it is."""
    counts = np.bincount(ytr, minlength=3)
    small = counts[counts > 0].min()
    if small >= 6:
        return SMOTE(k_neighbors=5, random_state=seed).fit_resample(Xtr, ytr)
    if small >= 2:
        return SMOTE(k_neighbors=small - 1, random_state=seed).fit_resample(Xtr, ytr)
    big = counts.max()
    strat = {c: big for c in range(3) if counts[c] >= 2}
    k = int(min(counts[c] for c in strat) - 1)
    return SMOTE(sampling_strategy=strat, k_neighbors=min(5, k), random_state=seed).fit_resample(Xtr, ytr)


def fit(est, Xtr, ytr, seed, Xval=None, yval=None):
    """Balance the TRAINING data only with SMOTE, then fit the model.
    Validation/test data are never oversampled (no leakage)."""
    if est == "xgb":
        if Xval is None:   # inside CV there is no separate validation set: carve 15% off the training fold
            strat = ytr if np.bincount(ytr).min() >= 2 else None
            Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.15, stratify=strat, random_state=seed)
        Xs, ys = smote(Xtr, ytr, seed)
        m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        m.fit(Xs, ys, eval_set=[(Xval, yval)], verbose=False)     # stops when validation log-loss stalls
        return m
    Xs, ys = smote(Xtr, ytr, seed)
    return est.fit(Xs, ys)


def metrics(y, proba, pred):
    """Macro metrics used in the thesis. Brier = mean squared error of the
    probability vector, divided by the number of classes (original convention)."""
    yb = label_binarize(y, classes=[0, 1, 2])
    return {"F1_macro": f1_score(y, pred, average="macro"),
            "AUC_ROC": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
            "AUC_PR": average_precision_score(yb, proba, average="macro"),
            "Brier": np.mean(np.sum((proba - yb) ** 2, axis=1)) / 3,
            "Accuracy": accuracy_score(y, pred)}


def split(y, seed, idx=None):
    """70/15/15 stratified split exactly as in the original run."""
    idx = np.arange(len(y)) if idx is None else idx
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y[idx], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, va, te


def mean_sd(df, by):
    g = df.groupby(by, sort=False)
    return g.mean(numeric_only=True).round(3).astype(str) + " ± " + g.std(numeric_only=True).round(3).astype(str)


def mean_abs_shap(values, n_feat):
    """Mean |SHAP| per feature, whatever array layout the SHAP version returns."""
    a = np.abs(np.array(values))
    fa = [i for i in range(a.ndim) if a.shape[i] == n_feat][-1]
    return a.mean(axis=tuple(i for i in range(a.ndim) if i != fa))


def run_split(X, y, tr, va, te, seed, names=None):
    """Fit the chosen models on one split; return {name: (proba, pred, model)}."""
    out = {}
    for name, est in models(seed).items():
        if names and name not in names:
            continue
        m = fit(est, X.iloc[tr], y[tr], seed, X.iloc[va], y[va]) if est == "xgb" else fit(est, X.iloc[tr], y[tr], seed)
        out[name] = (m.predict_proba(X.iloc[te]), m.predict(X.iloc[te]), m)
    return out


# ============================== MAIN ======================================
def main(path):
    os.makedirs(OUT, exist_ok=True)
    df = load(path)
    y = make_target(df[TARGET_SOURCE].values, THRESHOLDS_MAIN)
    X = encode(df)
    tier = df[TIER_COL].astype(str).values
    log = {"n_rows": len(df), "n_features": X.shape[1], "class_counts_LMH": np.bincount(y).tolist()}
    print("STEP 0:", X.shape, "classes", np.bincount(y))

    # ---------- STEP 1: original split + LightGBM (S3) -------------------
    tr, va, te = split(y, 42)
    res = run_split(X, y, tr, va, te, 42)
    rng = np.random.default_rng(42)
    rows, pc_rows = [], []
    for name in MODEL_ORDER:
        proba, pred, _ = res[name]
        boots = []
        for _ in range(N_BOOT):                      # resample the 45 test cases with replacement
            b = rng.integers(0, len(te), len(te))
            boots.append(f1_score(y[te][b], pred[b], average="macro", labels=[0, 1, 2], zero_division=0))
        k = int((pred == y[te]).sum())
        lo, hi = proportion_confint(k, len(te), method="wilson")
        rows.append({"model": name, **metrics(y[te], proba, pred), "F1_CI_low": np.percentile(boots, 2.5),
                     "F1_CI_high": np.percentile(boots, 97.5), "correct": k, "acc_CI_low": lo, "acc_CI_high": hi})
        pc_rows.append({"model": name, **dict(zip(["F1_Low", "F1_Medium", "F1_High"], f1_score(y[te], pred, average=None)))})
        pd.DataFrame(confusion_matrix(y[te], pred), index=CLASS_NAMES, columns=CLASS_NAMES)\
            .to_csv(f"{OUT}/S3_confusion_{name.split()[0]}.csv")
    pd.DataFrame(rows).round(3).to_csv(f"{OUT}/S3_test_metrics.csv", index=False)
    pd.DataFrame(pc_rows).round(3).to_csv(f"{OUT}/S3_per_class_f1.csv", index=False)
    # McNemar: b = only XGBoost right, c = only the baseline right
    rows = []
    xc = res["XGBoost (proposed)"][1] == y[te]
    for name in MODEL_ORDER[1:]:
        oc = res[name][1] == y[te]
        b, c = int((xc & ~oc).sum()), int((~xc & oc).sum())
        r = mcnemar([[0, b], [c, 0]], exact=True)
        rows.append({"comparison": f"XGBoost vs {name}", "b": b, "c": c, "statistic": r.statistic,
                     "p_value": round(r.pvalue, 4),
                     "cohens_g": round(max(b, c) / (b + c) - 0.5, 3) if b + c else 0.0,
                     "accuracy_diff": round((b - c) / len(te), 3)})
    pd.DataFrame(rows).to_csv(f"{OUT}/S3_mcnemar.csv", index=False)
    xgb_main = res["XGBoost (proposed)"][2]
    print("STEP 1 done")

    # ---------- STEP 2: 5-fold CV, all models (S1) ----------------------
    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (a, b) in enumerate(skf.split(tr, y[tr])):
        fa, fb = tr[a], tr[b]
        for name, est in models(42).items():
            m = fit(est, X.iloc[fa], y[fa], 42)
            rows.append({"model": name, "fold": k + 1, **metrics(y[fb], m.predict_proba(X.iloc[fb]), m.predict(X.iloc[fb]))})
    cv = pd.DataFrame(rows); cv.round(4).to_csv(f"{OUT}/S1_cv_folds.csv", index=False)
    mean_sd(cv.drop(columns="fold"), "model").to_csv(f"{OUT}/S1_cv_mean_sd.csv")
    print("STEP 2 done")

    # ---------- STEP 3: 10 seeds (S2) -----------------------------------
    rows, trees = [], []
    for s in SEEDS:
        t1, v1, e1 = split(y, s)
        r = run_split(X, y, t1, v1, e1, s)
        for name in MODEL_ORDER:
            rows.append({"model": name, "seed": s, **metrics(y[e1], r[name][0], r[name][1])})
        trees.append(r["XGBoost (proposed)"][2].best_iteration + 1)
    ms = pd.DataFrame(rows); ms.round(4).to_csv(f"{OUT}/S2_seed_runs.csv", index=False)
    mean_sd(ms.drop(columns="seed"), "model").to_csv(f"{OUT}/S2_seed_mean_sd.csv")
    ms.groupby("model", sort=False)["F1_macro"].agg(["min", "max", "std"]).round(3).to_csv(f"{OUT}/S2_seed_f1_range.csv")
    print("STEP 3 done")

    # Figure 5: stability box plot
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (d, title) in zip(axs, [(cv, "(a) 5-fold CV (training partition)"), (ms, "(b) 10 random seeds (held-out test)")]):
        data = [d.loc[d.model == m, "F1_macro"].values for m in MODEL_ORDER]
        ax.boxplot(data, showmeans=True, showfliers=False)
        ax.set_xticks(range(1, 6), [m.replace(" (proposed)", "\n(proposed)").replace(" ", "\n", 1) for m in MODEL_ORDER], fontsize=8)
        jr = np.random.default_rng(0)
        for i, v in enumerate(data):
            ax.scatter(jr.normal(i + 1, 0.04, len(v)), v, s=12, alpha=0.6, color="#2b4f8c", zorder=3)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("Macro F1-score")
    plt.tight_layout(); plt.savefig(f"{OUT}/Figure5_stability_boxplot.png", dpi=220); plt.close()

    # ---------- STEP 4: ablations (S4, S7, S8, S9) ------------------------
    Xc = encode(df, drop=INSTITUTIONAL)        # institutional covariates removed
    Xr = encode(df, collapse_rare=True)        # rare answer options merged
    y_alt = make_target(df[TARGET_SOURCE].values, THRESHOLDS_ALT)
    rows = []
    for s in SEEDS:
        t1, v1, e1 = split(y, s)
        for label, XX in [("Full model", X), ("No institutional covariates", Xc), ("Rare categories collapsed", Xr)]:
            m = fit("xgb", XX.iloc[t1], y[t1], s, XX.iloc[v1], y[v1])
            rows.append({"analysis": label, "p": XX.shape[1], "seed": s, **metrics(y[e1], m.predict_proba(XX.iloc[e1]), m.predict(XX.iloc[e1]))})
        # linear vs interaction logistic regression on the seven construct scores only
        C = X[CONSTRUCTS]
        for label, est in [("LR, F1-F7 additive", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
                           ("LR, F1-F7 + pairwise interactions", make_pipeline(StandardScaler(), PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)))]:
            m = fit(est, C.iloc[t1], y[t1], s)
            rows.append({"analysis": label, "p": 7 if "additive" in label else 28, "seed": s, **metrics(y[e1], m.predict_proba(C.iloc[e1]), m.predict(C.iloc[e1]))})
        # alternative thresholds: new target, new stratified split
        t2, v2, e2 = split(y_alt, s)
        m = fit("xgb", X.iloc[t2], y_alt[t2], s, X.iloc[v2], y_alt[v2])
        rows.append({"analysis": "Alternative thresholds (<=4 / 5-8 / >=9)", "p": 70, "seed": s, **metrics(y_alt[e2], m.predict_proba(X.iloc[e2]), m.predict(X.iloc[e2]))})
    ab = pd.DataFrame(rows); ab.round(4).to_csv(f"{OUT}/S4_S7_S8_S9_runs.csv", index=False)
    out = mean_sd(ab.drop(columns=["seed", "p"]), "analysis")
    out["p"] = ab.groupby("analysis", sort=False)["p"].first()
    out.to_csv(f"{OUT}/S4_S7_S8_S9_mean_sd.csv")
    log["alt_threshold_class_counts_LMH"] = np.bincount(y_alt).tolist()
    print("STEP 4 done")

    # ---------- STEP 5: leave-one-tier-out (S5) ---------------------------
    rows = []
    for t in sorted(set(tier)):
        trn, tst = np.where(tier != t)[0], np.where(tier == t)[0]
        for label, XX in [("Full model", X), ("No institutional covariates", Xc)]:
            m = fit("xgb", XX.iloc[trn], y[trn], 42)
            pred = m.predict(XX.iloc[tst])
            rows.append({"held_out_tier": t, "features": label, "n": len(tst),
                         "class_counts_LMH": "/".join(map(str, np.bincount(y[tst], minlength=3))),
                         "pred_counts_LMH": "/".join(map(str, np.bincount(pred, minlength=3))),
                         "accuracy": accuracy_score(y[tst], pred),
                         "F1_macro_present": f1_score(y[tst], pred, average="macro", labels=np.unique(y[tst]))})
    pd.DataFrame(rows).round(3).to_csv(f"{OUT}/S5_leave_one_tier_out.csv", index=False)
    print("STEP 5 done")

    # ---------- STEP 6: perturbation (S6) ---------------------------------
    expl = shap.TreeExplainer(xgb_main)
    Xte = X.iloc[te]
    base_pred = xgb_main.predict(Xte)
    base = pd.Series(mean_abs_shap(expl.shap_values(Xte), X.shape[1]), index=X.columns).sort_values(ascending=False)
    agree, rho, top5 = [], [], []
    for _ in range(NOISE_TRIALS):
        Xp = Xte.copy()
        for c in CONSTRUCTS:      # add measurement noise, keep inside the 1-5 Likert range
            Xp[c] = np.clip(Xp[c] + rng.normal(0, NOISE_SD, len(Xp)), 1, 5)
        agree.append((xgb_main.predict(Xp) == base_pred).mean())
        imp = pd.Series(mean_abs_shap(expl.shap_values(Xp), X.shape[1]), index=X.columns)
        rho.append(spearmanr(base.values, imp[base.index].values).correlation)
        top5.append(len(set(base.index[:5]) & set(imp.sort_values(ascending=False).index[:5])) / 5)
    pd.DataFrame([{"agreement_mean": np.mean(agree), "agreement_sd": np.std(agree), "agreement_min": np.min(agree),
                   "shap_rank_spearman_mean": np.mean(rho), "top5_overlap_mean": np.mean(top5)}]).round(3)\
        .to_csv(f"{OUT}/S6_perturbation.csv", index=False)
    print("STEP 6 done")

    # ---------- STEP 7: per-tier audit (S10) ------------------------------
    rows = []
    for t in sorted(set(tier[te])):
        mk = tier[te] == t
        for name in MODEL_ORDER:
            pr = res[name][1][mk]
            rows.append({"tier": t, "model": name, "n": int(mk.sum()),
                         "class_counts_LMH": "/".join(map(str, np.bincount(y[te][mk], minlength=3))),
                         "accuracy": accuracy_score(y[te][mk], pr),
                         "F1_macro_present": f1_score(y[te][mk], pr, average="macro", labels=np.unique(y[te][mk])),
                         "F1_macro_original": f1_score(y[te][mk], pr, average="macro")})
    pt = pd.DataFrame(rows).round(3); pt.to_csv(f"{OUT}/S10_per_tier.csv", index=False)
    pd.crosstab(tier, pd.Series(y).map(dict(enumerate(CLASS_NAMES))), margins=True)[["Low", "Medium", "High", "All"]]\
        .to_csv(f"{OUT}/S10_tier_by_class_full.csv")
    print("STEP 7 done")

    # ---------- STEP 8: complexity, SHAP, target proximity (S11-S13) -------
    tdf = xgb_main.get_booster().trees_to_dataframe()
    used = xgb_main.best_iteration + 1
    leaves = int(((tdf.Feature == "Leaf") & (tdf.Tree < used * 3)).sum())
    log.update({"xgb_rounds_used": used, "xgb_trees_total": used * 3, "xgb_leaves_total": leaves,
                "xgb_leaves_per_tree": round(leaves / (used * 3), 2),
                "xgb_rounds_across_seeds_mean": float(np.mean(trees)), "xgb_rounds_across_seeds_sd": float(np.std(trees))})
    sv = expl(Xte)
    imp = pd.Series(mean_abs_shap(sv.values, X.shape[1]), index=X.columns).sort_values(ascending=False)
    imp.round(4).to_csv(f"{OUT}/S12_shap_full.csv")
    short = lambda s: s if len(s) < 48 else s[:45] + "…"
    top = imp[:15][::-1]
    plt.figure(figsize=(8, 6)); plt.barh([short(i) for i in top.index], top.values, color="#3b6fb6")
    plt.xlabel("Mean |SHAP value| (averaged across classes)"); plt.tight_layout()
    plt.savefig(f"{OUT}/Figure6a_shap_importance.png", dpi=220); plt.close()
    sv_high = sv[:, :, 2]
    sv_high.feature_names = [short(f) for f in X.columns]
    plt.figure(); shap.plots.beeswarm(sv_high, max_display=15, show=False)
    plt.gcf().set_size_inches(9, 6.5); plt.tight_layout()
    plt.savefig(f"{OUT}/Figure6b_shap_beeswarm_high.png", dpi=220, bbox_inches="tight"); plt.close()
    pd.Series(sv.values[:, :, 2].mean(0), index=X.columns).round(4).to_csv(f"{OUT}/S12_shap_high_mean_signed.csv")
    # constructs-only model SHAP ranking
    mc = fit("xgb", Xc.iloc[tr], y[tr], 42, Xc.iloc[va], y[va])
    pd.Series(mean_abs_shap(shap.TreeExplainer(mc).shap_values(Xc.iloc[te]), Xc.shape[1]), index=Xc.columns)\
        .sort_values(ascending=False).round(4).to_csv(f"{OUT}/S12_shap_no_institutional.csv")
    rows = []
    for c in CATEGORICAL:
        tab = pd.crosstab(df[c].astype(str), y)
        v = np.sqrt(chi2_contingency(tab)[0] / (tab.values.sum() * (min(tab.shape) - 1))) if min(tab.shape) > 1 else 0
        rows.append({"item": c, "levels": tab.shape[0], "cramers_v_with_tier": round(v, 3)})
    pd.DataFrame(rows).sort_values("cramers_v_with_tier", ascending=False).to_csv(f"{OUT}/S13_target_proximity.csv", index=False)
    print("STEP 8 done")

    # ---------- STEP 9: duplicate profiles (S14) ---------------------------
    # Many institutions gave IDENTICAL answers. If a twin sits in training and
    # the other in test, the model is partly tested on data it has already seen.
    profile = pd.util.hash_pandas_object(X, index=False).values
    groups = pd.factorize(profile)[0]
    twin_in_train = np.isin(profile[te], profile[tr])
    log.update({"duplicate_rows": int(pd.Series(profile).duplicated().sum()),
                "unique_profiles": int(len(set(profile))),
                "test_rows_with_twin_in_train": int(twin_in_train.sum()),
                "profiles_with_conflicting_labels": int(pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].nunique().gt(1).sum())})
    rows = []
    unseen = ~twin_in_train
    for name in MODEL_ORDER:     # main-split performance on test rows WITHOUT a twin in training
        proba, pred, _ = res[name]
        rows.append({"evaluation": "Main split, unseen profiles only", "model": name, "n": int(unseen.sum()),
                     "F1_macro": f1_score(y[te][unseen], pred[unseen], average="macro"),
                     "Accuracy": accuracy_score(y[te][unseen], pred[unseen])})
    pd.DataFrame(rows).round(3).to_csv(f"{OUT}/S14_unseen_profiles.csv", index=False)
    rows = []
    for s in SEEDS[:5]:          # grouped CV: identical profiles always land in the same fold
        sgk = StratifiedGroupKFold(5, shuffle=True, random_state=s)
        for k, (a, b) in enumerate(sgk.split(X, y, groups)):
            for name, est in models(s).items():
                m = fit(est, X.iloc[a], y[a], s)
                rows.append({"model": name, "seed": s, "fold": k + 1, **metrics(y[b], m.predict_proba(X.iloc[b]), m.predict(X.iloc[b]))})
    gcv = pd.DataFrame(rows); gcv.round(4).to_csv(f"{OUT}/S14_grouped_cv_runs.csv", index=False)
    mean_sd(gcv.drop(columns=["seed", "fold"]), "model").to_csv(f"{OUT}/S14_grouped_cv_mean_sd.csv")
    rows = []
    for s in SEEDS[:5]:
        sgk = StratifiedGroupKFold(5, shuffle=True, random_state=s)
        for k, (a, b) in enumerate(sgk.split(Xc, y, groups)):
            m = fit("xgb", Xc.iloc[a], y[a], s)
            rows.append({"model": "XGBoost, no institutional covariates", **metrics(y[b], m.predict_proba(Xc.iloc[b]), m.predict(Xc.iloc[b]))})
    mean_sd(pd.DataFrame(rows), "model").to_csv(f"{OUT}/S14_grouped_cv_no_institutional.csv")
    print("STEP 9 done")

    # ---------- STEP 10: equalised-odds audit (S15) ------------------------
    # Every respondent gets exactly one out-of-fold prediction from the grouped
    # CV (seed 42), so the audit uses all 300 cases instead of 45.
    oof = np.full(len(y), -1)
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for a, b in sgk.split(X, y, groups):
        oof[b] = fit("xgb", X.iloc[a], y[a], 42).predict(X.iloc[b])
    rows = []
    for t in sorted(set(tier)):
        mk = tier == t
        row = {"tier": t, "n": int(mk.sum()), "class_counts_LMH": "/".join(map(str, np.bincount(y[mk], minlength=3))),
               "accuracy": accuracy_score(y[mk], oof[mk]),
               "F1_macro_present": f1_score(y[mk], oof[mk], average="macro", labels=np.unique(y[mk]))}
        for k, cname in enumerate(CLASS_NAMES):   # one-vs-rest true- and false-positive rates
            pos, neg = y[mk] == k, y[mk] != k
            row[f"TPR_{cname}"] = (oof[mk][pos] == k).mean() if pos.sum() else np.nan
            row[f"FPR_{cname}"] = (oof[mk][neg] == k).mean() if neg.sum() else np.nan
        rows.append(row)
    eo = pd.DataFrame(rows)
    gaps = {}
    for cname in CLASS_NAMES:   # largest between-tier gap, using only tiers with >= 5 positives
        ok = [r for r in rows if np.bincount(y[tier == r["tier"]], minlength=3)[CLASS_NAMES.index(cname)] >= 5]
        if len(ok) >= 2:
            gaps[f"TPR_gap_{cname}"] = max(r[f"TPR_{cname}"] for r in ok) - min(r[f"TPR_{cname}"] for r in ok)
            gaps[f"FPR_gap_{cname}"] = max(r[f"FPR_{cname}"] for r in ok) - min(r[f"FPR_{cname}"] for r in ok)
    eo.round(3).to_csv(f"{OUT}/S15_equalised_odds_by_tier.csv", index=False)
    pd.Series(gaps).round(3).to_csv(f"{OUT}/S15_equalised_odds_gaps.csv")
    log["oof_accuracy_all"] = float(accuracy_score(y, oof)); log["oof_f1_all"] = float(f1_score(y, oof, average="macro"))
    print("STEP 10 done")

    # ---------- STEP 11: SHAP vs gain; waterfalls (S16, S17) ----------------
    gain = pd.Series(xgb_main.get_booster().get_score(importance_type="gain")).reindex(X.columns).fillna(0)
    shap_imp = pd.Series(mean_abs_shap(sv.values, X.shape[1]), index=X.columns)
    cmp_df = pd.DataFrame({"mean_abs_shap": shap_imp, "gain": gain})
    cmp_df["shap_rank"] = cmp_df.mean_abs_shap.rank(ascending=False); cmp_df["gain_rank"] = cmp_df.gain.rank(ascending=False)
    cmp_df.sort_values("shap_rank").round(4).to_csv(f"{OUT}/S16_shap_vs_gain.csv")
    top_union = cmp_df.sort_values("shap_rank").index[:15]
    log["shap_gain_spearman_all"] = float(spearmanr(cmp_df.mean_abs_shap, cmp_df.gain).correlation)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    lbl = [short(i) for i in top_union][::-1]
    axs[0].barh(lbl, (cmp_df.loc[top_union, "mean_abs_shap"] / cmp_df.mean_abs_shap.max())[::-1], color="#3b6fb6")
    axs[0].set_title("(a) Mean |SHAP| (scaled to max = 1)")
    axs[1].barh(lbl, (cmp_df.loc[top_union, "gain"] / cmp_df.gain.max())[::-1], color="#dd8452")
    axs[1].set_title("(b) XGBoost gain (scaled to max = 1)")
    plt.tight_layout(); plt.savefig(f"{OUT}/FigureB3_shap_vs_gain.png", dpi=200); plt.close()
    proba_te = xgb_main.predict_proba(Xte); pred_te = proba_te.argmax(1)
    for k, cname in enumerate(CLASS_NAMES):   # most confident correct case of each tier
        cand = np.where((pred_te == k) & (y[te] == k))[0]
        if len(cand) == 0:
            continue
        i = cand[np.argmax(proba_te[cand, k])]
        e = sv[i, :, k]; e.feature_names = [short(f) for f in X.columns]
        plt.figure(); shap.plots.waterfall(e, max_display=10, show=False)
        plt.gcf().set_size_inches(8, 5); plt.title(f"{cname}-readiness case ({tier[te][i]})", fontsize=10)
        plt.savefig(f"{OUT}/FigureB4_waterfall_{cname}.png", dpi=200, bbox_inches="tight"); plt.close()
    print("STEP 11 done")
    json.dump(log, open(f"{OUT}/run_log.json", "w"), indent=2)
    print("All steps done ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="AI_Readiness_Cleaned_Workbook_300_1.xlsx")
    main(ap.parse_args().data)
