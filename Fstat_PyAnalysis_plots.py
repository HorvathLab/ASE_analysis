import os
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
import ptitprince as pt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

# === Load data ===
df = pd.read_csv("./concat_annot_distr_filtered_snp157_germ.txt", sep='\t')  # Change this filename as needed
print(df.head(2))
# Long reads: concat_annot_distr_filtered_snp157_germ.txt
# Short reads: concat_28_VAF5_95_snp157_germ.txt
# Long reads novel: concat_annot_distr_filtered_VAF5_95_novel.txt





df_columns = {
    "BayesDel_addAF": ("F", "p","BayesDel_addAF_score", "BayesDel_addAF_rankscore", "BayesDel_addAF_pred"),
    "SIFT": ("F", "p","SIFT_score", "SIFT_converted_rankscore", "SIFT_pred"),
    "SIFT4G": ("F", "p","SIFT4G_score", "SIFT4G_converted_rankscore", "SIFT4G_pred"),
    "MutationTaster": ("F", "p","MutationTaster_score", "MutationTaster_converted_rankscore", "MutationTaster_pred"),
    "MutationAssessor": ("F", "p","MutationAssessor_score", "MutationAssessor_rankscore", "MutationAssessor_pred"),
    "FATHMM": ("F", "p","FATHMM_score", "FATHMM_converted_rankscore", "FATHMM_pred"),
    "PROVEAN": ("F", "p","PROVEAN_score", "PROVEAN_converted_rankscore", "PROVEAN_pred"),
    "MetaSVM": ("F", "p","MetaSVM_score", "MetaSVM_rankscore", "MetaSVM_pred"),
    "MetaLR": ("F", "p","MetaLR_score", "MetaLR_rankscore", "MetaLR_pred"),
    "MetaRNN": ("F", "p","MetaRNN_score", "MetaRNN_rankscore", "MetaRNN_pred"),
    "M.CAP": ("F", "p","M.CAP_score", "M.CAP_rankscore", "M.CAP_pred"),
    "PrimateAI": ("F", "p","PrimateAI_score", "PrimateAI_rankscore", "PrimateAI_pred"),
    "DEOGEN2": ("F", "p","DEOGEN2_score", "DEOGEN2_rankscore", "DEOGEN2_pred"),
    "BayesDel_noAF": ("F", "p","BayesDel_noAF_score", "BayesDel_noAF_rankscore", "BayesDel_noAF_pred"),
    "ClinPred": ("F", "p","ClinPred_score", "ClinPred_rankscore", "ClinPred_pred"),
    "LIST.S2": ("F", "p","LIST.S2_score", "LIST.S2_rankscore", "LIST.S2_pred"),
    "fathmm.MKL_coding": ("F", "p","fathmm.MKL_coding_score", "fathmm.MKL_coding_rankscore", "fathmm.MKL_coding_pred"),
    "fathmm.XF_coding": ("F", "p","fathmm.XF_coding_score", "fathmm.XF_coding_rankscore", "fathmm.XF_coding_pred")
}

df.replace('.', pd.NA, inplace=True)

results = []
# Loop through each tool in df_columns
for tool, cols in df_columns.items():
    # Create output folder for this tool
    outdir = f'test/{tool}'
    os.makedirs(outdir, exist_ok=True)

    f_stat_col, p_col, score_col, rank_col, pred_col = cols

    df_subset = df[['F', 'p', score_col, rank_col, pred_col]].dropna()
    
    # Drop rows with missing values in any of the required columns
    df_subset[score_col] = pd.to_numeric(df_subset[score_col])
    df_subset[['F', score_col, pred_col]].dropna(inplace=True)

    # Compute F quantiles and assign groups
    q25 = df_subset['F'].quantile(0.25)
    q75 = df_subset['F'].quantile(0.75)
    median_F = df_subset['F'].median()


    # === Assign F-group labels ===
    def classify_f_group(f):
        if f <= q25:
            return 'Low F'
        elif f >= q75:
            return 'High F'
        else:
            return 'Mid'

    df_subset['F_group'] = df_subset['F'].apply(classify_f_group)


    df_subset['F_group'] = df_subset['F'].apply(
        lambda f: 'Low F' if f <= q25 else ('High F' if f >= q75 else 'Mid')
    )

    print(df_subset.head(2))    

    predictor_order = sorted([x for x in df_subset[pred_col].unique() if x != 'D']) + ['D']

    if tool=='LRT':
        df_subset = df_subset[df_subset[pred_col] != 'U']
        predictor_order = ['D', 'N']

    if tool=='MutationAssessor':
        df_subset = df_subset[df_subset[pred_col] != 'L']
        df_subset = df_subset[df_subset[pred_col] != 'M']
        predictor_order = ['H', 'N']
    
    if tool=='MutationTaster':
        df_subset[pred_col] = df_subset[pred_col].replace('A', 'D')
        df_subset[pred_col] = df_subset[pred_col].replace('P', 'N')
        predictor_order = ['D', 'N']

    custom_pal = {x: '#FC0353' if x == 'D' or x=='H' else '#4A90E2' for x in predictor_order}
    cust_pal = {
        'Low F': "#4A90E2",   
        'Mid': "#C53AD4",     
        'High F': "#FC0353"
    }   

    # === Kruskal-Wallis test ===
    groups = [df_subset[df_subset['F_group'] == grp][score_col] for grp in ['Low F', 'Mid', 'High F']]
    kw_stat, kw_p = kruskal(*groups)
    kw_text_f = f"Kruskal-Wallis p-value = {kw_p:.2e}"
    print(f"\nKruskal-Wallis Test Result: H = {kw_stat:.2f}, p = {kw_p:.2e}")

    results.append({
        "tool": tool,
        "test": "score_by_F_group",
        "statistic": float(kw_stat),
        "p_value": float(kw_p),
        "groups": "Low F|Mid|High",
        "n_per_group": "|".join(str(len(g)) for g in groups)
    })

    group_labels = df_subset[pred_col].unique()  
    group_labels = sorted(group_labels)
    groups = [df_subset[df_subset[pred_col] == label]['F'] for label in group_labels]

    # === Raincloud: F-stat by prediction ===
    # Prepare group labels and Kruskal-Wallis test
    group_labels = sorted(df_subset[pred_col].unique())
    groups = [df_subset[df_subset[pred_col] == label]['F'] for label in group_labels]
    kw_stat, kw_p = kruskal(*groups)
    kw_text = f"Kruskal-Wallis p = {kw_p:.2e}"

    results.append({
        "tool": tool,
        "test": "F_by_prediction",
        "statistic": float(kw_stat),
        "p_value": float(kw_p),
        "groups": "|".join(group_labels),
        "n_per_group": "|".join(str(len(g)) for g in groups)
    })
    
    # Set up plot
    plt.figure(figsize=(7, 6))
    ax = pt.RainCloud(
        x=pred_col,
        y='F',
        data=df_subset,
        palette=custom_pal,
        bw=0.3,
        width_viol=0.6,
        width_box=0.15,
        orient='v',
        move=0.0,                # nudges violins from axis
        alpha=1,
        dodge=True,
        order=group_labels,
        box_showfliers=False,
        point_size=3,
        linewidth=0.0,             # no outline on violin
    )

    # Labels and stats
    plt.xlabel("")
    plt.ylabel('F-stat', fontsize=14)
    ax.text(
        0.5, 0.93, kw_text,
        transform=ax.transAxes,
        ha='center', va='bottom', fontsize=11,
    )

    # Improve spacing
    plt.margins(x=0.1)
    plt.tight_layout()
    plt.savefig(f"{outdir}/raincloud_f_by_pred_with_points_generic.png", dpi=300)
    plt.close()


    group_stats = df_subset.groupby('F_group')[score_col].agg(['mean', 'median', 'std']).round(2)
    group_order = ['Low F', 'Mid', 'High F']

 
    group_stats = df_subset.groupby('F_group')[score_col].agg(['mean', 'median', 'std']).round(2)
    group_order = ['Low F', 'Mid', 'High F']

    

    

    # === Violin + Box of F-stat ===
    plt.figure(figsize=(6, 8))
    sns.violinplot(y=df_subset['F'], inner=None, color='lightgray')
    sns.boxplot(y=df_subset['F'], width=0.2, showcaps=True, showfliers=False,
                boxprops={'facecolor': 'white', 'edgecolor': 'black'},
                whiskerprops={'color': 'black'}, medianprops={'color': 'black'})
    plt.axhline(q25, color='blue', linestyle='--', linewidth=1.5, label=f'25th percentile: {q25:.3f}')
    plt.axhline(median_F, color='black', linestyle='--', linewidth=1.5, label=f'Median: {median_F:.3f}')
    plt.axhline(q75, color='red', linestyle='--', linewidth=1.5, label=f'75th percentile: {q75:.3f}')
    plt.title('F-statistics: Violin + Box Plot with Quantiles')
    plt.ylabel('F-stat')
    plt.xticks([])
    plt.grid(True, axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{outdir}/violin_box_fstat_quantiles_generic.png")
    plt.close()

    plt.figure(figsize=(6, 8))
    ax = plt.gca()

    # Violin + Boxplot
    sns.violinplot(y=df_subset['F'], inner=None, color='lightgray')
    sns.boxplot(
        y=df_subset['F'],
        width=0.2,
        showcaps=True,
        showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black'},
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'}
    )

    # Quantile lines
    plt.axhline(q25, color='blue', linestyle='--', linewidth=1.5, label=f'25th percentile: {q25:.3f}')
    plt.axhline(median_F, color='black', linestyle='--', linewidth=1.5, label=f'Median: {median_F:.3f}')
    plt.axhline(q75, color='red', linestyle='--', linewidth=1.5, label=f'75th percentile: {q75:.3f}')

    # === Add mean/median/std annotation ===
    mean_f = df_subset['F'].mean()
    std_f = df_subset['F'].std()
    stats_text = f"Mean: {mean_f:.2f}\nMedian: {median_F:.2f}\nStd: {std_f:.2f}"
    # Final formatting
    plt.title('F-statistics: Violin + Box Plot with Quantiles')
    plt.ylabel('F-stat')
    plt.xticks([])
    plt.grid(False, axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{outdir}/violin_box_fstat_quantiles_generic.png")
    plt.close()


    # Plot
    plt.figure(figsize=(8, 6))
    #sns.set(style="whitegrid", font_scale=1.4, rc={"axes.labelsize": 16, "axes.titlesize": 18})
    ax = pt.RainCloud(
        x='F_group',
        y=score_col,
        data=df_subset,
        palette=cust_pal,
        bw=.3,
        width_viol=.6,
        width_box=.2,
        orient='v',
        move=0.0,
        alpha=1,
        dodge=True,
        order=['Low F', 'Mid', 'High F'],
        box_showfliers=False,
        point_size=3,
        linewidth=0  # Remove outline
    )

    # Title, labels, stats
    plt.xlabel("Quantile Group", fontsize=16)
    plt.ylabel('F-stat Group', fontsize=16)
    ax.text(
        0.5, 0.93, kw_text_f,
        transform=ax.transAxes,
        ha='center', va='bottom', fontsize=11,
    )
    y_min, y_max = df_subset[score_col].min(), df_subset[score_col].max()
    y_range = y_max - y_min

    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_xlim(-0.6, len(group_order) - 0.4)
    plt.tight_layout()
    plt.savefig(f"{outdir}/raincloud_score_by_fgroup_generic.png")
    plt.close()



    # Correlation plots
    g = sns.jointplot(data=df_subset, x=score_col, y='F', kind='reg', height=6, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    # Set x and y labels using the JointGrid's ax_joint
    g.ax_joint.set_xlabel("Predictor Score", fontsize=14)
    g.ax_joint.set_ylabel("F", fontsize=14)
    plt.savefig(f"{outdir}/CORR_jointplot_sift_vs_f.png")
    plt.close()

    corr = df_subset[['F', score_col]].corr()
    print(f"Plots saved for {tool} in {outdir}")

# === Summarize to one row per tool ===
summary = (
    pd.DataFrame(results)
      .pivot_table(index="tool", columns="test", values="p_value", aggfunc="first")
      .rename(columns={
          "score_by_F_group": "KW_test_by_quantile_group",
          "F_by_prediction": "KW_test_by_predictor"
      })
      .reset_index()
      .rename(columns={"tool": "Tool"})
)

# Format p-values as .2e
summary["KW_test_by_quantile_group"] = summary["KW_test_by_quantile_group"].map(lambda x: f"{x:.2e}")
summary["KW_test_by_predictor"] = summary["KW_test_by_predictor"].map(lambda x: f"{x:.2e}")


os.makedirs("test", exist_ok=True)
summary.to_csv("test/kruskal_pvalues_summary_wide_short.csv", index=False)
print("Saved: test/kruskal_pvalues_summary_wide.csv")

