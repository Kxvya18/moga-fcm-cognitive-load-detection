










import os, random, warnings
import numpy as np
import pandas as pd
from deap import base, creator, tools
import skfuzzy as fuzz
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, pairwise_distances)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")




INPUT_CSV     = "affective_normalized.csv"
OUT_ROOT      = "ga_results"
DRIVER_SUBSET = [1, 3, 4, 5, 6]

MASK_MIN = 8
MASK_MAX = 25
N_RANDOM = 20

NGEN_VALUES = [90]
POP_VALUES  = [150]




print(f"Loading {INPUT_CSV} ...")
df        = pd.read_csv(INPUT_CSV)
meta_cols = ["driver_id", "window_start", "gt_label"]
features  = [c for c in df.columns if c not in meta_cols]
x_raw     = np.clip(df[features].values.astype(np.float64), 0.0, 1.0)
driver_id = df["driver_id"].values.astype(int)

var  = x_raw.var(axis=0)
keep = var > 1e-8
x_raw    = x_raw[:, keep]
features = [f for f, k in zip(features, keep) if k]

n_features     = x_raw.shape[1]
active_drivers = np.array([d for d in np.unique(driver_id) if d in DRIVER_SUBSET])

print(f"Features : {n_features}")
print(f"Windows  : {len(df)}")
print(f"Drivers  : {active_drivers.tolist()}")




MRT_MAP = {
    "hrv_sdnn":           2.0, "hrv_rmssd":          2.0,
    "hrv_sd1":            2.0, "hrv_sd2":            2.0,
    "hrv_pnn50":          2.0, "hrv_sample_entropy": 2.0,
    "hrv_stress_index":   2.0, "hrv_lf_hf":          1.9,
    "hrv_hf_power":       1.9, "hrv_lf_power":       1.8,
    "hrv_total_power":    1.8, "eda_scr_count":      2.0,
    "eda_scr_amp":        2.0, "eda_scr_rate":       1.9,
    "eda_phasic_power":   1.9, "eda_scl_mean":       1.8,
    "eda_phasic_std":     1.7, "bvp_lf_power":       1.8,
    "bvp_hf_power":       1.8, "bvp_lf_hf":          1.8,
    "bvp_hr_std":         1.7, "bvp_rr_std":         1.7,
    "bvp_sample_entropy": 1.6, "cross_sympathetic":  1.8,
    "cross_autonomic":    1.7, "cross_eda_hr":       1.7,
    "e4_hr_std":          1.5, "e4_hr_rmssd":        1.5,
    "bio_hr_rmssd":       1.5, "bio_hr_cv":          1.4,
    "bio_br_mean":        1.4, "bio_br_std":         1.4,
    "temp_slope":         1.3, "temp_std":           1.2,
    "acc_enmo":           1.2, "acc_jerk":           1.1,
    "posture":            1.0, "activity":           1.0,
}

def mrt_weight(fname):
    fl = fname.lower()
    for kw, w in MRT_MAP.items():
        if fl.startswith(kw) or kw in fl:
            return w
    return 1.0

feature_mrt_w  = np.array([mrt_weight(f) for f in features])
feature_mrt_w /= (feature_mrt_w.mean() + 1e-9)
print(f"MRT weight range: [{feature_mrt_w.min():.2f}, {feature_mrt_w.max():.2f}]")





def cluster_metrics(xsel, labels):
    uniq  = np.unique(labels)
    cents = np.array([xsel[labels == k].mean(axis=0) for k in uniq])
    wcd   = np.mean([np.mean(cdist(xsel[labels == k], [cents[i]]))
                     for i, k in enumerate(uniq)])
    bcd   = float(np.mean(pairwise_distances(cents)))
    return float(wcd), bcd

def _sil(X, lb):
    u = np.unique(lb)
    if len(u) < 2 or X.shape[0] <= len(u): return 0.0
    return float(silhouette_score(X, lb))

def _dbi(X, lb):
    u = np.unique(lb)
    if len(u) < 2 or X.shape[0] <= len(u): return None
    return float(davies_bouldin_score(X, lb))

def _chi(X, lb):
    u = np.unique(lb)
    if len(u) < 2 or X.shape[0] <= len(u): return None
    return float(calinski_harabasz_score(X, lb))





if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness,
                   weights=(-1.0, 1.0, 1.0, -1.0, 1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("gene",       random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate",   tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
ref_pts = tools.uniform_reference_points(nobj=5, p=12)
toolbox.register("select", tools.selNSGA3, ref_points=ref_pts)





def evaluate(ind, xtrain, mrt_w):
    mask  = np.array(ind, dtype=bool)
    nfeat = int(mask.sum())
    if nfeat < MASK_MIN or nfeat > MASK_MAX:
        return nfeat, 0.0, 0.0, 1e6, 0.0
    xsel = np.clip(xtrain[:, mask], 0.0, 1.0).astype(np.float64)
    try:
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            xsel.T, c=3, m=2.0, error=0.005, maxiter=1000, init=None)
        labels = np.argmax(u, axis=0)
    except Exception:
        return nfeat, 0.0, 0.0, 1e6, 0.0
    uniq = np.unique(labels)
    if len(uniq) < 2 or xsel.shape[0] <= len(uniq):
        return nfeat, 0.0, 0.0, 1e6, 0.0
    try:
        raw_mi = mutual_info_classif(xsel, labels, discrete_features=False)
        wmi    = float(np.mean(raw_mi * mrt_w[mask]))
    except Exception:
        wmi = 0.0
    sil = float(np.nan_to_num(silhouette_score(xsel, labels), nan=0.0))
    wcd, bcd = cluster_metrics(xsel, labels)
    if not np.isfinite(wcd): wcd = 1e6
    if not np.isfinite(bcd): bcd = 0.0
    penalty = len(uniq) / (xsel.shape[0] + 1e-9)
    overlap = 1.0 / (bcd + 1e-9)
    sil_adj = sil / (1.0 + 0.5 * penalty + 0.1 * overlap)
    return nfeat, wmi, sil_adj, wcd, bcd





def run_ga(xtrain, driver_val, ngen, pop_size, mrt_w, cx=0.8, mu=0.2):
    dim = xtrain.shape[1]
    for reg in ["individual", "population", "evaluate"]:
        try: toolbox.unregister(reg)
        except Exception: pass
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.gene, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, xtrain=xtrain, mrt_w=mrt_w)

    pop  = toolbox.population(n=pop_size)
    hof  = tools.HallOfFame(10)
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hof.update(pop)

    gen_log = []
    for gen in range(1, ngen + 1):
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(pop, 2)
            c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
            if random.random() < cx:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
            if random.random() < mu:
                toolbox.mutate(c1); del c1.fitness.values
            if random.random() < mu:
                toolbox.mutate(c2); del c2.fitness.values
            offspring.extend([c1, c2])
        offspring = offspring[:pop_size]
        invalid   = [ind for ind in offspring if not ind.fitness.valid]
        fits      = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, k=pop_size)
        hof.update(pop)

        fa = np.array([ind.fitness.values for ind in pop])
        nc, wc, sc, wdc, bc = fa[:,0], fa[:,1], fa[:,2], fa[:,3], fa[:,4]
        pf = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        ps = [ind.fitness.values[2] for ind in pf]
        pn = [int(sum(ind)) for ind in pf]

        gen_log.append({
            "gen":  gen, "n_evals": len(invalid),
            "nfeat_min":  float(nc.min()),  "nfeat_mean": float(nc.mean()), "nfeat_max": float(nc.max()),
            "wmi_max":    float(wc.max()),   "wmi_mean":   float(wc.mean()), "wmi_min":  float(wc.min()),
            "sil_max":    float(sc.max()),   "sil_mean":   float(sc.mean()), "sil_std":  float(sc.std()),
            "wcd_min":    float(wdc[wdc < 1e5].min())  if (wdc < 1e5).any() else 1e6,
            "wcd_mean":   float(wdc[wdc < 1e5].mean()) if (wdc < 1e5).any() else 1e6,
            "bcd_max":    float(bc.max()),   "bcd_mean":   float(bc.mean()),
            "pareto_size":       len(pf),
            "pareto_sil_max":    float(max(ps))     if ps else 0.0,
            "pareto_sil_mean":   float(np.mean(ps)) if ps else 0.0,
            "pareto_nfeat_min":  int(min(pn))       if pn else 0,
            "pareto_nfeat_mean": float(np.mean(pn)) if pn else 0.0,
            "hof_best_sil":      float(hof[0].fitness.values[2]) if hof else 0.0,
            "hof_best_nfeat":    int(sum(hof[0]))                if hof else 0,
            "hof_best_wmi":      float(hof[0].fitness.values[1]) if hof else 0.0,
            "hof_best_bcd":      float(hof[0].fitness.values[4]) if hof else 0.0,
        })
        if gen % 10 == 0 or gen == ngen:
            print(f"    gen={gen:3d} | pareto={len(pf):3d} | "
                  f"sil_max={sc.max():.3f} | nfeat_mean={nc.mean():.1f} | "
                  f"wmi_max={wc.max():.4f}")

    pf = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pf, hof, gen_log


def save_artefacts(pareto, hof, gen_log, drv, feats_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ind in pareto:
        mask = np.array(ind, dtype=int)
        sel  = [f for f, m in zip(feats_list, mask) if m]
        rows.append({
            "driver":      drv,
            "mask":        mask.tolist(),
            "n_features":  int(mask.sum()),
            "f1_nfeat":    ind.fitness.values[0],
            "f2_wmi":      ind.fitness.values[1],
            "f3_sil_adj":  ind.fitness.values[2],
            "f4_wcd":      ind.fitness.values[3],
            "f5_bcd":      ind.fitness.values[4],
            "selected_features": ";".join(sel),
        })
    pd.DataFrame(rows).to_csv(f"{out_dir}/driver_{drv}_pareto.csv", index=False)

    hof_rows = []
    for rank, ind in enumerate(hof):
        mask = np.array(ind, dtype=int)
        sel  = [f for f, m in zip(feats_list, mask) if m]
        hof_rows.append({
            "rank": rank, "driver": drv, "mask": mask.tolist(),
            "n_features": int(mask.sum()),
            "f1_nfeat":   ind.fitness.values[0], "f2_wmi":  ind.fitness.values[1],
            "f3_sil_adj": ind.fitness.values[2], "f4_wcd":  ind.fitness.values[3],
            "f5_bcd":     ind.fitness.values[4],
            "selected_features": ";".join(sel),
        })
    pd.DataFrame(hof_rows).to_csv(f"{out_dir}/driver_{drv}_hof.csv", index=False)
    pd.DataFrame(gen_log).to_csv(f"{out_dir}/driver_{drv}_convergence.csv", index=False)

    snap = [{"driver": drv, "n_features": int(sum(ind)),
             "sil_adj": ind.fitness.values[2], "bcd": ind.fitness.values[4],
             "wcd":     ind.fitness.values[3], "wmi": ind.fitness.values[1]}
            for ind in pareto]
    pd.DataFrame(snap).to_csv(f"{out_dir}/driver_{drv}_pareto_scatter.csv", index=False)
    print(f"    Saved -> {out_dir}/driver_{drv}_*")






def baseline_pca_fcm(xtrain, xtest, var=0.95, k=3, m=2.0):

    pca = PCA(n_components=var, svd_solver='full')
    pca.fit(xtrain)
    xs = pca.transform(xtest)
    try:
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            xs.T, c=k, m=m, error=0.005, maxiter=1000, init=None)
        lb = np.argmax(u, axis=0)
    except Exception:
        return {"sil": 0.0, "dbi": None, "chi": None, "n_comp": xs.shape[1]}
    return {"sil": _sil(xs, lb), "dbi": _dbi(xs, lb),
            "chi": _chi(xs, lb), "n_comp": int(xs.shape[1])}


def baseline_gmm(xtest, k=3):

    try:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=42, n_init=5, max_iter=200)
        lb  = gmm.fit_predict(xtest)
    except Exception:
        return {"sil": 0.0, "dbi": None, "chi": None}
    return {"sil": _sil(xtest, lb), "dbi": _dbi(xtest, lb), "chi": _chi(xtest, lb)}


def baseline_agglomerative(xtest, k=3):

    try:
        lb = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(xtest)
    except Exception:
        return {"sil": 0.0, "dbi": None, "chi": None}
    return {"sil": _sil(xtest, lb), "dbi": _dbi(xtest, lb), "chi": _chi(xtest, lb)}


def baseline_random_fcm(xtest, n=N_RANDOM, k=3, m=2.0):

    results = []
    nf = xtest.shape[1]
    for seed in range(n):
        rng = np.random.RandomState(seed)
        nf2 = rng.randint(MASK_MIN, MASK_MAX + 1)
        idx = rng.choice(nf, size=min(nf2, nf), replace=False)
        xs  = xtest[:, idx]
        try:
            _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                xs.T, c=k, m=m, error=0.005, maxiter=1000, init=None)
            lb = np.argmax(u, axis=0)
        except Exception:
            results.append(0.0); continue
        u2 = np.unique(lb)
        results.append(_sil(xs, lb) if len(u2) > 1 and xs.shape[0] > len(u2) else 0.0)
    return {"sil_mean": float(np.mean(results)),
            "sil_std":  float(np.std(results)),
            "sil_max":  float(np.max(results))}





os.makedirs(OUT_ROOT, exist_ok=True)

feat_freq     = np.zeros(n_features, dtype=int)
baseline_rows = []
grid_summary  = []

for i in active_drivers:
    print(f"\n{'='*65}\n  DRIVER {i}  --  LODO\n{'='*65}")
    test_idx  = np.where(driver_id == i)[0]
    train_idx = np.where(driver_id != i)[0]
    if len(test_idx) < 10 or len(train_idx) < 20:
        print(f"  Skip (test={len(test_idx)} train={len(train_idx)})"); continue
    xtrain, xtest = x_raw[train_idx], x_raw[test_idx]


    bl_pca   = baseline_pca_fcm(xtrain, xtest)
    bl_gmm   = baseline_gmm(xtest)
    bl_agglo = baseline_agglomerative(xtest)
    bl_rnd   = baseline_random_fcm(xtest)

    baseline_rows.append({
        "driver":       int(i),
        "pca_fcm_sil":  bl_pca["sil"],  "pca_fcm_dbi": bl_pca.get("dbi"),
        "pca_fcm_chi":  bl_pca.get("chi"), "pca_n_comp": bl_pca.get("n_comp"),
        "gmm_sil":      bl_gmm["sil"],  "gmm_dbi":     bl_gmm.get("dbi"),
        "gmm_chi":      bl_gmm.get("chi"),
        "agglo_sil":    bl_agglo["sil"], "agglo_dbi":   bl_agglo.get("dbi"),
        "agglo_chi":    bl_agglo.get("chi"),
        "rand_sil_mean":bl_rnd["sil_mean"], "rand_sil_std": bl_rnd["sil_std"],
        "rand_sil_max": bl_rnd["sil_max"],
    })
    print(f"  Baselines ->  PCA+FCM={bl_pca['sil']:.3f} | "
          f"GMM={bl_gmm['sil']:.3f} | "
          f"Agglo={bl_agglo['sil']:.3f} | "
          f"Random={bl_rnd['sil_mean']:.3f}+-{bl_rnd['sil_std']:.3f}")


    for ngen in NGEN_VALUES:
        for pop_size in POP_VALUES:
            tag     = f"ngen{ngen}_pop{pop_size}"
            out_dir = f"{OUT_ROOT}/{tag}"
            print(f"\n  -- NGEN={ngen} POP={pop_size} --")
            pareto, hof, gen_log = run_ga(xtrain, i, ngen, pop_size, feature_mrt_w)
            save_artefacts(pareto, hof, gen_log, i, features, out_dir)
            best_sil = max((ind.fitness.values[2] for ind in pareto), default=0.0)
            grid_summary.append({
                "driver":   int(i), "ngen": ngen, "pop": pop_size,
                "pareto_n": len(pareto),
                "best_sil": round(best_sil, 4),
                "hof_sil":  round(hof[0].fitness.values[2], 4) if hof else 0.0,
            })
            top20 = sorted(pareto, key=lambda ind: ind.fitness.values[2], reverse=True)[:20]
            for ind in top20:
                feat_freq += np.array(ind, dtype=int)





fi = pd.DataFrame({
    "feature":    features,
    "frequency":  feat_freq,
    "mrt_weight": [mrt_weight(f) for f in features],
    "score":      feat_freq * np.array([mrt_weight(f) for f in features]),
}).sort_values("score", ascending=False)
fi.to_csv(f"{OUT_ROOT}/global_feature_importance.csv", index=False)
print(f"\n=== Top 20 Features ===\n{fi.head(20).to_string(index=False)}")

if baseline_rows:
    bl = pd.DataFrame(baseline_rows)
    bl.to_csv(f"{OUT_ROOT}/baseline_summary.csv", index=False)
    print(f"\n=== Baselines (mean +- std across {len(bl)} drivers) ===")
    print(f"  PCA+FCM  SIL: {bl['pca_fcm_sil'].mean():.3f} +- {bl['pca_fcm_sil'].std():.3f}")
    print(f"  GMM      SIL: {bl['gmm_sil'].mean():.3f} +- {bl['gmm_sil'].std():.3f}")
    print(f"  Agglo    SIL: {bl['agglo_sil'].mean():.3f} +- {bl['agglo_sil'].std():.3f}")
    print(f"  Random   SIL: {bl['rand_sil_mean'].mean():.3f} +- {bl['rand_sil_std'].mean():.3f}")

if grid_summary:
    gd = pd.DataFrame(grid_summary)
    gd.to_csv(f"{OUT_ROOT}/ga_grid_summary.csv", index=False)
    print(f"\n=== GA Summary ===")
    print(gd.pivot_table(values="best_sil", index=["ngen", "pop"],
                         aggfunc="mean").to_string())

print(f"\nAll outputs -> {OUT_ROOT}/\nNext: python part3_affectiveroad.py")
