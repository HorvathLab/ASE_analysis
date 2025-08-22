import os
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import statsmodels.stats.proportion as smp
import statsmodels.graphics.mosaicplot as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

file_path = "./concat_annot_distr_filtered_snp157_germ.txt"
save_path = "./test_Z"
count=1
kruskal_csv = "test/kruskal_pvalues_summary_wide_short.csv"
# Long reads: concat_annot_distr_filtered_snp157_germ.txt
# Short reads: concat_28_VAF5_95_snp157_germ.txt
# Long reads novel: concat_annot_distr_filtered_VAF5_95_novel.txt



# Create rules for each functionality column

functionality_rules = {
    "SIFT_pred": ("SIFT_pred_Functionality", 
                  lambda df: df[df["SIFT_pred"].isin(["D", "T"])],
                  lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "SIFT4G_pred": ("SIFT4G_pred_Functionality", 
                  lambda df: df[df["SIFT4G_pred"].isin(["D", "T"])],
                  lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "MutationTaster_pred": ("MutationTaster_pred_Functionality",
                            lambda df: df[df["MutationTaster_pred"].isin(["D", "P", "N"])],
                            lambda x:"Functional" if x == 'D' else "Non-Functional"),
    "MutationAssessor_pred":("MutationAssessor_pred_Functionality",
                             lambda df: df[df["MutationAssessor_pred"].isin(["H", "N"])],
                             lambda x:"Functional" if x == 'H' else "Non-Functional"),
    "FATHMM_pred":("FATHMM_pred_Functionality",
                   lambda df: df[df["FATHMM_pred"].isin(["D", "T"])],
                   lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "PROVEAN_pred":("PROVEAN_pred_Functionality",
                    lambda df: df[df["PROVEAN_pred"].isin(["D", "N"])],
                    lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "MetaSVM_pred":("MetaSVM_pred_Functionality",
                    lambda df: df[df["MetaSVM_pred"].isin(["D", "T"])],
                    lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "MetaLR_pred":("MetaLR_pred_Functionality",
                   lambda df: df[df["MetaLR_pred"].isin(["D", "T"])],
                   lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "MetaRNN_pred":("MetaRNN_pred_Functionality",
                   lambda df: df[df["MetaRNN_pred"].isin(["D", "T"])],
                   lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "M.CAP_pred":("M.CAP_pred_Functionality",
                  lambda df: df[df["M.CAP_pred"].isin(["D", "T"])],
                  lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "PrimateAI_pred":("PrimateAI_pred_Functionality",
                      lambda df: df[df["PrimateAI_pred"].isin(["D", "T"])],
                      lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "DEOGEN2_pred":("DEOGEN2_pred__Functionality",
                    lambda df: df[df["DEOGEN2_pred"].isin(["D", "T"])],
                    lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "BayesDel_addAF_pred":("BayesDel_addAF_pred_Functionality",
                           lambda df: df[df["BayesDel_addAF_pred"].isin(["D", "T"])],
                           lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "BayesDel_noAF_pred":("BayesDel_noAF_pred_Functionality",
                          lambda df: df[df["BayesDel_noAF_pred"].isin(["D", "T"])],
                          lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "ClinPred_pred":("ClinPred_pred_Functionality",
                     lambda df: df[df["ClinPred_pred"].isin(["D", "T"])],
                     lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "LIST.S2_pred":("LIST.S2_pred_Functionality",
                    lambda df: df[df["LIST.S2_pred"].isin(["D", "T"])],
                    lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "fathmm.MKL_coding_pred":("fathmm.MKL_coding_pred_Functionality",
                              lambda df: df[df["fathmm.MKL_coding_pred"].isin(["D", "N"])],
                              lambda x: "Functional" if x == 'D' else "Non-Functional"),
    "fathmm.XF_coding_pred":("fathmm.XF_coding_pred_Functionality",
                             lambda df: df[df["fathmm.XF_coding_pred"].isin(["D", "N"])],
                             lambda x: "Functional" if x == 'D' else "Non-Functional")
}


pred_to_tool = {
    "SIFT_pred": "SIFT",
    "SIFT4G_pred": "SIFT4G",
    "MutationTaster_pred": "MutationTaster",
    "MutationAssessor_pred": "MutationAssessor",
    "FATHMM_pred": "FATHMM",
    "PROVEAN_pred": "PROVEAN",
    "MetaSVM_pred": "MetaSVM",
    "MetaLR_pred": "MetaLR",
    "MetaRNN_pred": "MetaRNN",
    "M.CAP_pred": "M.CAP",
    "PrimateAI_pred": "PrimateAI",
    "DEOGEN2_pred": "DEOGEN2",
    "BayesDel_addAF_pred": "BayesDel_addAF",
    "BayesDel_noAF_pred": "BayesDel_noAF",
    "ClinPred_pred": "ClinPred",
    "LIST.S2_pred": "LIST.S2",
    "fathmm.MKL_coding_pred": "fathmm.MKL_coding",
    "fathmm.XF_coding_pred": "fathmm.XF_coding",
}
ztest_rows = []
# Loop through each column and perform the two proportion Z test for each column with p-value

for col, (new_col, filter, map) in functionality_rules.items():
    
    results = []

    data = pd.read_table(file_path, sep = "\t",low_memory=False)
    print(count,"Statistical test for",col)
    data=filter(data)
    data[new_col] = data[col].apply(map)
    threshold = 0.05
    data["functionGVS_threshold"] = data["p"].apply(lambda x:"Significant" if x <= threshold else "Non-Significant")
    
    # Creating a contigency table
    contingency_table = pd.crosstab(data[new_col],data["functionGVS_threshold"])
    new_order = ['Significant', 'Non-Significant']
    contingency_table = contingency_table[new_order]

    # Convert the contingency table to the desired format
    contingency_table_df = contingency_table.stack().reset_index()
    contingency_table_df.columns = ['Func_vs_nonFunc', 'functionGVS_threshold', 'count']

    # Create a PrettyTable
    table = PrettyTable()

    # Set the field names (columns)
    table.field_names = [contingency_table.index.name or ""] + list(contingency_table.columns)

    # Add rows
    for idx, row in contingency_table.iterrows():
        table.add_row([idx] + list(row.values))

    # Print the formatted table
    print(table)
    print("\n")

    #print(contingency_table_df)

    # Extract the counts
    successes = [contingency_table.iloc[0,0], contingency_table.iloc[1,0]]
    totals = [sum(contingency_table.iloc[0]), sum(contingency_table.iloc[1])]  
    non_functional_proportion = contingency_table.iloc[1,0]/sum(contingency_table.iloc[1])
    functional_proportion = contingency_table.iloc[0,0]/sum(contingency_table.iloc[0])
            
    print("Functional significant proportion: ",round(functional_proportion*100,2),"%")
    print("Non functional significant proportion: ",round(non_functional_proportion*100,2),"%")

    # Perform the two-proportion z-test

    z_stat, p_value = smp.proportions_ztest(successes, totals, alternative='larger')
    kw_text = f"Z-Test p-value: {p_value:.2e}"
    print("p_value: ",p_value)
    if p_value<=0.05:
        print("Sample is significant")
    else:
        print("Sample is not significant")

    # Append results to a tuple

    results.append((round(z_stat,4),round(p_value,4)))
    print("\n")

    # Convert results to dataframe and print

    contingency_results = pd.DataFrame(results)
    contingency_results.columns = ["Z_statistic", "p_value"]
    print(contingency_results)
    print("\n")

    count+=1

    # generate csv
    # Ensure all categories exist in a fixed order
    contingency_table = contingency_table.reindex(index=['Functional', 'Non-Functional'], columns=['Significant', 'Non-Significant'], fill_value=0)

    # Extract counts
    func_sig = int(contingency_table.loc['Functional', 'Significant'])
    func_nonsig = int(contingency_table.loc['Functional', 'Non-Significant'])
    nonfunc_sig = int(contingency_table.loc['Non-Functional', 'Significant'])
    nonfunc_nonsig = int(contingency_table.loc['Non-Functional', 'Non-Significant'])

    total_sig = func_sig + nonfunc_sig
    total_all = func_sig + func_nonsig + nonfunc_sig + nonfunc_nonsig
    prop_sig_overall = total_sig / total_all if total_all > 0 else np.nan  # overall proportion of SNVs with significant ASE

    # Format p-value as .2e for the CSV
    p_val_fmt = f"{p_value:.2e}"

    # Append one row for this predictor/tool
    ztest_rows.append({
        "Tool": pred_to_tool.get(col, col),
        "Z_test_p_value_by_pred": p_val_fmt,
        "Proportion of SNVs with significant ASE - D": f"{round(functional_proportion*100, 2)}%",
        "Proportion of SNVs with significant ASE - T": f"{round(non_functional_proportion*100, 2)}%",

        # Contingency values for mosaic plot (explicit columns)
        "Functional_Significant": func_sig,
        "Functional_Non_Significant": func_nonsig,
        "Non_Functional_Significant": nonfunc_sig,
        "Non_Functional_Non_Significant": nonfunc_nonsig,
    })


    # Visualization of the results

    ## Create a bar plot to show proportion of functional and non functional values
    
    fig, ax = plt.subplots(figsize=(5, 4))

    categories = ["Non-Functional", "Functional"]
    per = [non_functional_proportion * 100, functional_proportion * 100]

    bars = ax.bar(categories, per,color=['#4A90E2', '#FC0353'],width=0.5)

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{round(height,2)}%', ha='center', va='bottom')

    # Convert p-value to stars
    if p_value <= 0.0001:
        stars = "****"
    elif p_value <= 0.001:
        stars = "***"
    elif p_value <= 0.01:
        stars = "**"
    elif p_value <= 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Get positions for significance bracket
    max_y = max(per)
    bar_x_positions = [bar.get_x() + bar.get_width()/2 for bar in bars]

    # Add bracket
    bracket_height = max_y + 14
    line_height = max_y + 12

    # Horizontal line
    ax.hlines(y=line_height, xmin=bar_x_positions[0], xmax=bar_x_positions[1], 
            color='black', linewidth=1)

    # Vertical lines
    ax.vlines(x=bar_x_positions[0], ymin=line_height-3, ymax=line_height, 
            color='black', linewidth=1)
    ax.vlines(x=bar_x_positions[1], ymin=line_height-3, ymax=line_height, 
            color='black', linewidth=1)

    # Add stars
    ax.text((bar_x_positions[0] + bar_x_positions[1])/2, bracket_height,
            stars, ha='center', va='bottom', fontsize=14)
    
    # Add p-value
    ax.text(
        0.5, 0.93, kw_text,
        transform=ax.transAxes,
        ha='center', va='bottom', fontsize=11,
    )

    # Customize layout
    ax.set_xticks([0,1])   
    ax.set_xticklabels(['T', 'D'])
    ax.set_yticklabels('')
    ax.set_ylabel('Proportion of SNVs with significant ASE')

    # Set y-axis range and percentage format
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + f"/Bar_plot{col}.png")




    ## Create a mosaic plot for contingency table

    fig, ax2 = plt.subplots(figsize=(7, 4))

    # Create a dictionary for the mosaic plot
    mosaic_data = {(row['Func_vs_nonFunc'], row['functionGVS_threshold']): row['count'] for index, row in contingency_table_df.iterrows()}

    #Custom labelizer function to include counts
    def label_with_count(key):
        return f"{mosaic_data[key]}"

    # Define colors for each main category
    colors = {
        ('Functional', 'Significant'): '#FC0353',
        ('Functional', 'Non-Significant'): '#FC0353',
        ('Non-Functional', 'Significant'): '#4A90E2', 
        ('Non-Functional', 'Non-Significant'): '#4A90E2'
    }

    # Create the mosaic plot

    #fig_1, _ = mpl.mosaic(mosaic_data, title='Two Proportion Z-test Contingency Table', labelizer=label_with_count,properties = lambda key: {'color': colors[key]},gap=0.01,ax=ax2)
    fig_1, _ = mpl.mosaic(mosaic_data,labelizer=label_with_count,properties = lambda key: {'color': colors[key]},gap=0.01,ax=ax2)

    # Create labels
    ax2.set_xticklabels([])
    ax2.tick_params(axis='x', bottom=False)
    ax2.set_yticklabels(["Skewed","Random"])
    plt.savefig(save_path + f"/Mosaic_plot{col}.png")


    ## Violin plot of p-values by SNV group

    fig, ax3 = plt.subplots(figsize=(5, 4))

    data['Func_vs_nonFunc'] = pd.Categorical(data[new_col], categories=['Non-Functional', 'Functional'], ordered=True)
    violin_plot = sns.violinplot(x=data["Func_vs_nonFunc"],y=data["p"],ax=ax3,linewidth=1)


    from matplotlib.collections import PolyCollection

    # Get all violin body collections (skip boxes/lines)
    violin_bodies = [c for c in ax3.collections if isinstance(c, PolyCollection)]

    # Assuming order = ['Non-Functional', 'Functional']
    # The first violin is at index 0, second at index 1
    non_func_violin = violin_bodies[0]
    func_violin = violin_bodies[1]

    non_func_violin.set_facecolor('#4A90E2')  # Blue
    func_violin.set_facecolor('#FC0353')      # Red
    non_func_violin.set_edgecolor('none')
    func_violin.set_edgecolor('none')
    
    violin_plot.set_xticks([0,1])
    violin_plot.set_xticklabels(['T', 'D'], fontsize=14)
    # ax3.set_yticklabels("")
    violin_plot.set_xlabel("")
    violin_plot.set_ylabel("Significant values proportion")

    # Add annotations
    x1, x2 = 0, 1  
    y = data["p"].max() * 1.1  
    kw_text = f"p-value: {p_value:.2e}"
    ax3.text((x1 + x2) / 2, 0.5, stars, ha="center", va="bottom",fontsize = 14)
    ax3.text(
        0.5, 0.85, kw_text,
        transform=ax.transAxes,
        ha='center', va='bottom', fontsize=11,
    )
    ax3.text(
        0.5, 0.80, "Z-Test",
        transform=ax.transAxes,
        ha='center', va='bottom', fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path + f"/Violin_plot{col}.png")
    plt.close()


# === Build Z-test/contingency summary and merge with Kruskal CSV ===
ztest_df = pd.DataFrame(ztest_rows)

# Read the wide Kruskal summary
kruskal_df = pd.read_csv(kruskal_csv)
# Reformat Kruskal p-values to .2e after reading
for col in ["KW_test_by_quantile_group", "KW_test_by_predictor"]:
    kruskal_df[col] = kruskal_df[col].map(lambda x: f"{x:.2e}" if pd.notnull(x) else "")

# Merge on Tool
merged = kruskal_df.merge(ztest_df, on="Tool", how="left")

# Save both individual and merged outputs
ztest_out = os.path.join(save_path, "ztest_contingency_summary.csv")
merged_out = os.path.join(save_path, "combined_stats_summary_short.csv")

ztest_df.to_csv(ztest_out, index=False)
merged.to_csv(merged_out, index=False)

print(f"Saved: {ztest_out}")
print(f"Saved: {merged_out}")
