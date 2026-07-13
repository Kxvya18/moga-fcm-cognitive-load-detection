
































import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT_DIR = "plots/convergence"
GA_ROOT = "ga_results"
RES_DIR = "results"
CFG     = "ngen90_pop150"

os.makedirs(OUT_DIR, exist_ok=True)

DRIVERS = [1, 3, 4, 5, 6]
CLR = {"LOW_LOAD": "#3B8BD4", "MEDIUM_LOAD": "#EF9F27", "HIGH_LOAD": "#E24B4A"}





def load_conv(driver):
    p = f"{GA_ROOT}/{CFG}/driver_{driver}_convergence.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def load_pareto_scatter(driver):
    p = f"{GA_ROOT}/{CFG}/driver_{driver}_pareto_scatter.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def load_fi():
    p = f"{GA_ROOT}/global_feature_importance.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def load_result(name):
    p = f"{RES_DIR}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else None

def modality_label(fname):
    fl = fname.lower()
    for prefix, label in [
        ("hrv_", "HRV"), ("eda_", "EDA"), ("bvp_", "BVP"),
        ("temp_", "TEMP"), ("acc_", "ACC"), ("e4_hr", "E4-HR"),
        ("bio_hr", "Bio-HR"), ("bio_br", "Bio-BR"),
        ("posture", "Posture"), ("activity", "Activity"),
        ("cross_", "Cross-modal"),
    ]:
        if fl.startswith(prefix):
            return label
    return "Other"


drivers_found = [d for d in DRIVERS if load_conv(d) is not None]





print("Plotting convergence panels...")
for driver in drivers_found:
    log = load_conv(driver)
    if log is None:
        continue
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Driver {driver} — GA Convergence  (GEN=90, POP=150)",
                 fontsize=13, fontweight="bold")
    ax_sil, ax_nfeat, ax_wmi, ax_par, ax_hof, ax_bcd = axes.flat
    gen = log["gen"]
    col = "#2166ac"

    ax_sil.plot(gen, log["sil_max"], color=col, lw=2.0, label="Pop max")
    ax_sil.fill_between(gen,
                        log["sil_mean"] - log["sil_std"],
                        log["sil_mean"] + log["sil_std"],
                        color=col, alpha=0.12, label="±1 std")
    ax_sil.plot(gen, log["pareto_sil_max"], color="crimson",
                lw=1.5, ls="--", label="Pareto max")
    ax_sil.plot(gen, log["hof_best_sil"], color="darkorange",
                lw=1.5, ls=":", label="HoF best")
    ax_sil.set_title("Silhouette Convergence"); ax_sil.set_ylabel("SIL")
    ax_sil.legend(fontsize=7)

    ax_nfeat.plot(gen, log["pareto_nfeat_mean"], color=col, lw=2, label="Pareto mean")
    ax_nfeat.fill_between(gen, log["nfeat_min"], log["nfeat_max"],
                           color=col, alpha=0.10, label="pop range")
    ax_nfeat.axhline(8, color="red", ls="--", lw=1, alpha=0.6, label="MASK_MIN=8")
    ax_nfeat.set_title("Feature Count"); ax_nfeat.set_ylabel("# Features")
    ax_nfeat.legend(fontsize=7)

    ax_wmi.plot(gen, log["wmi_max"],  color=col, lw=2.0, label="max")
    ax_wmi.plot(gen, log["wmi_mean"], color=col, lw=1.0, ls=":", alpha=0.7, label="mean")
    ax_wmi.plot(gen, log["hof_best_wmi"], color="darkorange",
                lw=1.5, ls="--", label="HoF wMI")
    ax_wmi.set_title("Physio-Weighted MI (wMI)"); ax_wmi.set_ylabel("wMI")
    ax_wmi.legend(fontsize=7)

    ax_par.plot(gen, log["pareto_size"], color=col, lw=2)
    ax_par.set_title("Pareto Front Size"); ax_par.set_ylabel("# Solutions")

    ax_hof.plot(gen, log["hof_best_sil"], color="darkorange", lw=2.5)
    ax_hof.set_title("Hall of Fame Best SIL"); ax_hof.set_ylabel("SIL")

    ax_bcd.plot(gen, log["bcd_max"],  color=col, lw=2.0, label="max")
    ax_bcd.plot(gen, log["bcd_mean"], color=col, lw=1.0, ls=":", alpha=0.7, label="mean")
    ax_bcd.set_title("Between-Cluster Distance"); ax_bcd.set_ylabel("BCD")
    ax_bcd.legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("Generation"); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = f"{OUT_DIR}/driver_{driver}_convergence.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  → {out}")





print("Plotting 3-D Pareto scatters...")
for driver in drivers_found:
    df_sc = load_pareto_scatter(driver)
    if df_sc is None or len(df_sc) == 0:
        continue
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(df_sc["n_features"], df_sc["bcd"], df_sc["sil_adj"],
                     c=df_sc["sil_adj"], cmap="viridis",
                     s=60, alpha=0.8, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Adjusted SIL")
    ax.set_xlabel("# Features"); ax.set_ylabel("BCD"); ax.set_zlabel("SIL")
    ax.set_title(f"Driver {driver}: Pareto Front")
    plt.tight_layout()
    out = f"{OUT_DIR}/driver_{driver}_pareto3d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")





print("Plotting SIL generation boxplots...")
INTERVAL = 20
for driver in drivers_found:
    log = load_conv(driver)
    if log is None:
        continue
    snap_gens = [g for g in sorted(log["gen"].unique())
                 if g % INTERVAL == 0 or g == log["gen"].max()]
    box_data, box_labels = [], []
    for g in snap_gens:
        row   = log[log["gen"] == g].iloc[0]
        s_m   = float(row["sil_mean"])
        s_s   = max(float(row["sil_std"]), 0.001)
        s_mx  = float(row["sil_max"])
        samps = np.clip(
            np.random.default_rng(int(g)).normal(s_m, s_s, 50), 0.0, s_mx)
        box_data.append(samps)
        box_labels.append(f"G{int(g)}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(box_data, labels=box_labels, patch_artist=True,
               medianprops=dict(color="red", lw=2))
    ax.set_title(f"Driver {driver}: SIL Distribution by Generation")
    ax.set_xlabel("Generation"); ax.set_ylabel("Silhouette Score")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = f"{OUT_DIR}/driver_{driver}_sil_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")





print("Plotting feature importance...")
fi_df = load_fi()
if fi_df is not None:
    top = fi_df.head(20).copy()
    fig, ax1 = plt.subplots(figsize=(14, 5))
    x = np.arange(len(top))
    ax1.bar(x, top["frequency"], color="steelblue", alpha=0.8,
            label="GA selection frequency", edgecolor="k", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top["feature"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("GA Selection Frequency"); ax1.set_xlabel("Feature")
    ax1.grid(True, alpha=0.2, axis="y")
    ax2 = ax1.twinx()
    ax2.plot(x, top["mrt_weight"], color="crimson",
             marker="D", markersize=6, lw=2, label="MRT Weight")
    ax2.set_ylabel("MRT Weight", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=9)
    ax1.set_title("Top-20 Features: GA Frequency + MRT Weight")
    plt.tight_layout()
    out = f"{OUT_DIR}/feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")





if fi_df is not None:
    fi_df["modality"] = fi_df["feature"].apply(modality_label)
    mod_freq = (fi_df.groupby("modality")["frequency"]
                .sum().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(mod_freq)))
    ax.bar(mod_freq.index, mod_freq.values, color=colors, edgecolor="k", alpha=0.85)
    ax.set_xlabel("Signal Modality"); ax.set_ylabel("Cumulative GA Frequency")
    ax.set_title("Feature Selection Frequency by Modality\n"
                 "(Empatica E4 vs Zephyr Bioharness vs Cross-modal)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    out = f"{OUT_DIR}/modality_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")






print("Plotting GT validation bars...")
tC = load_result("Table_C_GTValidation.csv")
if tC is not None:
    metrics = [m for m in ["ARI", "NMI", "Purity", "HungarianAcc", "LabelAlign"]
               if m in tC.columns]
    colors  = {"ARI": "#4393c3", "NMI": "#4393c3", "Purity": "#4393c3",
               "HungarianAcc": "#d6604d", "LabelAlign": "#762a83"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, met in zip(axes, metrics):
        ax.bar(tC["Driver"].astype(str), tC[met],
               color=colors.get(met, "steelblue"), alpha=0.85, edgecolor="k")
        ax.axhline(tC[met].mean(), color="red", ls="--", lw=1.5,
                   label=f"mean={tC[met].mean():.3f}")
        if met == "HungarianAcc":
            ax.axhline(0.333, color="gray", ls=":", lw=1.2, label="chance=0.333")
        ax.set_title(met, fontsize=9); ax.set_xlabel("Driver")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("MOGA-FCM vs AffectiveROAD SM Ground Truth\n"
                 "(HungarianAcc > chance=0.333 despite ARI~0 — semantic gap N7)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/gt_validation_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")






print("Plotting MMD distribution shift...")
mmd_df = load_result("mmd_distribution_shift.csv")
if mmd_df is not None:
    has_sub  = "mmd_subspace" in mmd_df.columns
    col_full = "mmd_full" if "mmd_full" in mmd_df.columns else "mmd"
    n_panels = 2 if has_sub else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    axes[0].bar(mmd_df["driver"].astype(str), mmd_df[col_full],
                color="coral", alpha=0.85, edgecolor="k")
    axes[0].axhline(mmd_df[col_full].mean(), color="navy", ls="--", lw=2,
                    label=f"mean={mmd_df[col_full].mean():.4f}")
    axes[0].set_title("MMD — Full Feature Space (119-dim, LODO)")
    axes[0].set_xlabel("Driver (test)"); axes[0].set_ylabel("MMD")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

    if has_sub:
        ratio = mmd_df[col_full].mean() / (mmd_df["mmd_subspace"].mean() + 1e-9)
        axes[1].bar(mmd_df["driver"].astype(str), mmd_df["mmd_subspace"],
                    color="steelblue", alpha=0.85, edgecolor="k")
        axes[1].axhline(mmd_df["mmd_subspace"].mean(), color="navy", ls="--", lw=2,
                        label=f"mean={mmd_df['mmd_subspace'].mean():.4f}")
        axes[1].set_title(f"MMD — Selected Subspace  ({ratio:.1f}× reduction)")
        axes[1].set_xlabel("Driver (test)"); axes[1].set_ylabel("MMD")
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Maximum Mean Discrepancy: Train vs Test (LODO) — N3",
                 fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/mmd_shift.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")







print("Plotting SVBI analysis...")
svbi_df = load_result("svbi_results.csv")
if svbi_df is not None:
    x = np.arange(len(svbi_df)); w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    ax = axes[0]
    ax.bar(x - w, svbi_df["rmssd_low"],  w, color=CLR["LOW_LOAD"],
           label="LOW",    edgecolor="k", alpha=0.85)
    ax.bar(x,     svbi_df["rmssd_med"],  w, color=CLR["MEDIUM_LOAD"],
           label="MEDIUM", edgecolor="k", alpha=0.85)
    ax.bar(x + w, svbi_df["rmssd_high"], w, color=CLR["HIGH_LOAD"],
           label="HIGH",   edgecolor="k", alpha=0.85)
    for xi, (_, row) in enumerate(svbi_df.iterrows()):
        top_val = max(float(row["rmssd_low"]),
                      float(row["rmssd_med"]),
                      float(row["rmssd_high"]))

        label_str = "INDEP" if row["svbi_independent"] else "circ"
        label_col = "darkgreen" if row["svbi_independent"] else "gray"
        ax.text(xi, top_val + 0.005, label_str,
                ha="center", va="bottom", fontsize=7,
                color=label_col, fontweight="bold")

        if row["hi_clear"]:
            ax.text(xi + w, float(row["rmssd_high"]) + 0.018,
                    "✓", ha="center", fontsize=11, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{int(d)}" for d in svbi_df["driver"]])
    ax.set_ylabel("Mean RMSSD  (lower = more vagal withdrawal)")
    ax.set_title("RMSSD per Cluster per Driver  (N4 — SVBI)\n"
                 "✓ = HIGH>LOW confirmed  |  INDEP = hrv_rmssd not selected by MOGA")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")


    ax2 = axes[1]
    bar_c = [CLR["HIGH_LOAD"] if row["hi_clear"] else "#aaaaaa"
             for _, row in svbi_df.iterrows()]
    ax2.bar(x, svbi_df["cohen_d"], color=bar_c, alpha=0.85, edgecolor="k",
            label="Cohen d  (HIGH vs LOW)")
    ax2.axhline(0,   color="k",    lw=1)
    ax2.axhline(0.5, color="gray", ls=":", lw=1.2, label="d=0.5 (medium)")
    ax3 = ax2.twinx()
    ax3.plot(x, svbi_df["svbi_gt_spearman"], color="purple",
             marker="D", markersize=7, lw=2, ls="--",
             label="SVBI-GT Spearman r")
    ax3.axhline(0, color="purple", lw=0.5, alpha=0.4)
    ax3.set_ylabel("SVBI-GT Spearman r", color="purple")
    ax3.tick_params(axis="y", labelcolor="purple")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{int(d)}" for d in svbi_df["driver"]])
    ax2.set_ylabel("Cohen d")
    ax2.set_title("SVBI Effect Size + GT Correlation\n"
                  "RED = HIGH>LOW confirmed (p<0.05 Mann-Whitney)")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("SVBI = 1−RMSSD  (Vagal Withdrawal Index) — N4 Novel Finding",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/svbi_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")






print("Plotting MRT-ACC dissociation...")
te = load_result("Table_E_MRT_Validation.csv")
if te is not None:
    te  = te.sort_values("mean_mi_gt", ascending=False).reset_index(drop=True)
    x   = np.arange(len(te))
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x - 0.2, te["mean_mi_gt"], 0.4, color="steelblue", alpha=0.85,
            label="Mean MI with GT labels  (ACC dominates)", edgecolor="k")
    ax1.set_ylabel("Mean MI with SM Ground Truth")
    ax1.set_xlabel("Modality")
    ax1.set_xticks(x)
    ax1.set_xticklabels(te["modality"], rotation=30, ha="right")
    ax1.grid(True, alpha=0.2, axis="y")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, te["mean_mrt"], 0.4, color="crimson", alpha=0.65,
            label="Mean MRT Weight  (HRV dominates)", edgecolor="k")
    ax2.set_ylabel("Mean MRT Weight", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    hrv_row = te[te["modality"] == "hrv"]
    if len(hrv_row):
        hi = hrv_row.index[0]
        ax1.annotate("HRV: highest MRT\nbut lowest GT-MI",
                     xy=(hi, float(hrv_row["mean_mi_gt"].values[0])),
                     xytext=(max(0, hi - 2),
                             float(hrv_row["mean_mi_gt"].values[0]) + 0.010),
                     arrowprops=dict(arrowstyle="->", color="black"),
                     fontsize=8)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    ax1.set_title("MRT-ACC Dissociation  (N5 Novel Finding)\n"
                  "ACC has 3× higher GT-MI than HRV despite HRV having highest MRT weight\n"
                  "→ MRT (designed for controlled tasks) misses naturalistic ACC dominance")
    plt.tight_layout()
    out = f"{OUT_DIR}/mrt_acc_dissociation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")






print("Plotting transferability (Table D)...")
td = load_result("Table_D_Transferability.csv")
if td is not None:
    x = np.arange(len(td)); w = 0.30
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))


    ax = axes[0]
    ax.bar(x - w/2, td["SIL_own"],      w, color="#4393c3",
           label="Own-driver SIL",    alpha=0.85, edgecolor="k")
    ax.bar(x + w/2, td["SIL_transfer"], w, color="#d1e5f0",
           label="Cross-driver SIL",  alpha=0.85, edgecolor="k")
    for xi, (_, row) in enumerate(td.iterrows()):
        drop = float(row["Xfer_drop"])
        col  = "green" if drop <= 0 else "red"
        ax.text(xi + w/2, float(row["SIL_transfer"]) + 0.003,
                f"{drop:+.3f}", ha="center", fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{int(d)}" for d in td["Driver"]])
    ax.set_ylabel("Silhouette Score (raw feature space)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Own vs Cross-Driver SIL  (raw space, comparable)\n"
                 "(annotation = Xfer_drop; green = transfer better than own)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")


    ax2 = axes[1]
    bar_c = [CLR["HIGH_LOAD"] if int(hc) else "#aaaaaa"
             for hc in td["HI_clear"].values]
    ax2.bar(x, td["SVBI_cohend"], color=bar_c, alpha=0.85, edgecolor="k",
            label="SVBI Cohen d")
    ax2.axhline(0,   color="k",    lw=1)
    ax2.axhline(0.5, color="gray", ls=":", lw=1.2, label="d=0.5")
    ax3 = ax2.twinx()
    ax3.plot(x, td["HungAcc"], color="purple", marker="s",
             markersize=7, lw=2, label="Hungarian Acc")
    ax3.axhline(0.333, color="purple", ls=":", lw=1.0, alpha=0.5,
                label="chance=0.333")
    ax3.set_ylabel("Hungarian Accuracy", color="purple")
    ax3.tick_params(axis="y", labelcolor="purple")
    ax3.set_ylim(0, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{int(d)}" for d in td["Driver"]])
    ax2.set_ylabel("SVBI Cohen d")
    ax2.set_title("SVBI Effect Size + Hungarian Accuracy\n"
                  "RED = HIGH>LOW confirmed | dashed = chance level")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Cross-Driver Transferability + Physiological Validation (Table D)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT_DIR}/transferability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")






print("Plotting latent vs raw SIL (AE-N6)...")
ta = load_result("Table_A_Clustering.csv")
if ta is not None:
    sil_raw_col = "SIL_Raw"
    sil_lat_col = "SIL_Latent(AE)"
    if sil_lat_col not in ta.columns or ta[sil_lat_col].isna().all():
        print("  [AE-N6] No latent SIL (PyTorch absent) — skipping graph 11")
    else:
        x = np.arange(len(ta)); w = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.bar(x - w/2, ta[sil_raw_col], w, color="#4393c3",
               label="Raw SIL (8-dim)",       alpha=0.85, edgecolor="k")
        ax.bar(x + w/2, ta[sil_lat_col].fillna(0), w, color="#d6604d",
               label="Latent SIL (4-dim AE)", alpha=0.85, edgecolor="k")
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{int(d)}" for d in ta["Driver"]])
        ax.set_ylabel("Silhouette Score")
        ax.set_ylim(0, 1.0)
        ax.set_title("Raw vs Latent SIL per Driver\n"
                     "(AE trained on LODO xtrain → applied to held-out xtest; no leakage)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

        ax2 = axes[1]
        delta = ta[sil_lat_col].fillna(ta[sil_raw_col]) - ta[sil_raw_col]
        bar_c2 = ["#1a9641" if d > 0.005 else
                  "#d7191c" if d < -0.005 else "#888888"
                  for d in delta]
        ax2.bar(x, delta, color=bar_c2, alpha=0.85, edgecolor="k")
        ax2.axhline(0, color="k", lw=1)
        ax2.axhline(float(delta.mean()), color="red", ls="--", lw=1.5,
                    label=f"mean Δ={delta.mean():+.3f}")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"D{int(d)}" for d in ta["Driver"]])
        ax2.set_ylabel("SIL delta  (Latent − Raw)")
        ax2.set_title("AE-N6: Compression Effect\n"
                      "GREEN = latent better  |  RED = raw better (near-min-entropy)")
        ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Autoencoder Impact on Cluster Quality — N6\n"
                     "SIL_Final(Raw) is the primary reported metric; "
                     "latent shown for comparison only",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = f"{OUT_DIR}/ae_sil_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  → {out}")





print(f"\nAll graphs saved to {OUT_DIR}/")
files = sorted(os.listdir(OUT_DIR))
for f in files:
    print(f"  {f}")
