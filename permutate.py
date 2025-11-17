import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import os

# -----------------------
# Load and preprocess data
# -----------------------
df = pd.read_csv("concat_annot_distr_filtered_snp157_germ.txt", sep="\t")
# novel: concat_annot_distr_filtered_VAF5_95_novel.txt
# longreads_ germ: concat_annot_distr_filtered_snp157_germ.txt
# 28smpls: 28smpls_predictor.txt


# Make sure all relevant columns exist
df.columns = df.columns.str.strip()

# Convert p, F, R2 columns to numeric
for col in ["p", "F", "R2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

saved_path = "plots_germ"
os.makedirs(saved_path, exist_ok=True)
# Change color palette for plots
color_palette = "#a2cf19"

# Predictor columns to iterate over
predictors = [
    "BayesDel_addAF_pred", "SIFT_pred", "SIFT4G_pred", "MutationTaster_pred",
    "MutationAssessor_pred", "FATHMM_pred", "PROVEAN_pred", "MetaSVM_pred",
    "MetaLR_pred", "MetaRNN_pred", "M.CAP_pred", "PrimateAI_pred", 
    "DEOGEN2_pred", "BayesDel_noAF_pred", "ClinPred_pred", "LIST.S2_pred",
    "fathmm.MKL_coding_pred", "fathmm.XF_coding_pred"
]

# -----------------------
# Define permutation test
# -----------------------
def permutation_test(x, y, n_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    observed = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        x_perm = combined[:len(x)]
        y_perm = combined[len(x):]
        stat = np.mean(x_perm) - np.mean(y_perm)
        if abs(stat) >= abs(observed):
            extreme += 1
    return observed, (extreme + 1) / (n_permutations + 1)

# -----------------------
# Run tests for all predictors
# -----------------------
results = []

for pred in predictors:
    if pred not in df.columns:
        print(f"Skipping {pred} (not found in file)")
        continue

    # Clean predictor column
    df[pred] = df[pred].astype(str).str.strip().str.upper()
    df = df[~df[pred].isin([".", "L", "M", "NAN"])].copy()

    # Recode if needed: H→D, N→T (for consistency)
    df[pred] = df[pred].replace({"H": "D", "N": "T"})

    # Filter only D and T rows
    sub = df[df[pred].isin(["D", "T"])].copy()
    if sub.empty:
        print(f"Skipping {pred}: no D/T data after filtering")
        continue

    # Extract arrays
    p_D = sub.loc[sub[pred] == "D", "p"].astype(float).values
    p_T = sub.loc[sub[pred] == "T", "p"].astype(float).values

    f_D = sub.loc[sub[pred] == "D", "F"].astype(float).values
    f_T = sub.loc[sub[pred] == "T", "F"].astype(float).values

    # Skip if any group is empty
    if len(p_D) == 0 or len(p_T) == 0:
        print(f"Skipping {pred}: one group empty")
        continue

    # --- Run permutation tests ---
    obs_p, perm_p = permutation_test(p_D, p_T)
    obs_f, perm_f = permutation_test(f_D, f_T)

    # --- Run standard tests ---
    t_pstat, t_pval = ttest_ind(p_D, p_T, equal_var=False)
    u_pstat, mw_pval = mannwhitneyu(p_D, p_T, alternative="two-sided")

    # Save results
    results.append({
        "Predictor": pred,
        "n_D": len(p_D),
        "n_T": len(p_T),
        "ObsDiff_p": obs_p,
        "Perm_pval_p": perm_p,
        "ObsDiff_F": obs_f,
        "Perm_pval_F": perm_f
    })

    # --- Visualization 1: histogram of p-values ---
    plt.figure(figsize=(7,5))
    plt.hist(p_D, bins=30, alpha=0.6, label="D", color="skyblue")
    plt.hist(p_T, bins=30, alpha=0.6, label="T", color="orange")
    plt.xlabel("p value")
    plt.ylabel("Count")
    plt.title(f"{pred}: Distribution of p-values (D vs T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{saved_path}/{pred}_pvalue_hist.png", dpi=300)
    plt.close()

    # --- Visualization 2: permutation null distribution ---
    perm_stats = []
    combined = np.concatenate([p_D, p_T])
    rng = np.random.default_rng(42)
    for _ in range(1000):
        rng.shuffle(combined)
        stat = np.mean(combined[:len(p_D)]) - np.mean(combined[len(p_D):])
        perm_stats.append(stat)
    plt.figure(figsize=(7,5))
    plt.hist(perm_stats, bins=50, alpha=0.7, color=color_palette, edgecolor="white", linewidth=0.7)
    plt.axvline(obs_p, color="red", linestyle="--", linewidth=2,
                label=f"Observed = {obs_p:.3f}")
    leg = plt.legend(
        [f"Observed = {obs_p:.3f}"],
        loc="upper right",
        frameon=False  # <--- removes legend box
    )
    plt.title(f"{pred}: Permutation Null Distribution (D vs T, p-values)")
    plt.xlabel("Mean Difference (D - T)")
    plt.ylabel("Frequency")
    plt.text(
        0.78, 0.92,
        f"p-value = {perm_p:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        verticalalignment="top"
    )
    plt.tight_layout()
    plt.savefig(f"{saved_path}/{pred}_permutation_pvalues.png", dpi=300)
    plt.close()

    # Visualizations 3: permutation null distribution for F-values
    perm_stats_f = []
    combined_f = np.concatenate([f_D, f_T])
    for _ in range(1000):
        rng.shuffle(combined_f)
        stat = np.mean(combined_f[:len(f_D)]) - np.mean(combined_f[len(f_D):])
        perm_stats_f.append(stat)
    plt.figure(figsize=(7,5))
    plt.hist(perm_stats_f, bins=50, alpha=0.7, color=color_palette, edgecolor="white", linewidth=0.7)
    # plt.axvline(obs_f, color="red", linestyle="--", linewidth=2,
    #             label=f"Observed = {obs_f:.3f}")
    # --- Cut the red line ---
    ymin, ymax = plt.gca().get_ylim()
    cutoff = ymax * 0.88  # stop at 80% of the height
    plt.vlines(obs_f, ymin, cutoff, color="red", linestyle="--", linewidth=2)

    leg = plt.legend(
        [f"Observed = {obs_f:.3f}"],
        loc="upper right",
        frameon=False  # <--- removes legend box
    )
    plt.title(f"{pred}: Permutation Null Distribution (D vs T, F-values)")
    plt.xlabel("Mean Difference (D - T)")
    plt.ylabel("Frequency")
    plt.text(
        0.78, 0.92,
        f"p-value = {perm_f:.4f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        verticalalignment="top"
    )
    plt.tight_layout()
    plt.savefig(f"{saved_path}/{pred}_permutation_fvalues.png", dpi=300)
    plt.close()     

# -----------------------
# Save results to file
# -----------------------
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by="Perm_pval_p", ascending=True)
res_df.to_csv(f"{saved_path}/predictor_permutation_results.csv", index=False)

