# coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

segmentations_df = pd.read_csv("segmentations.csv")
segmentations_df.loc[segmentations_df["Method"] == "AttentionUNet", "Method"] = "Pre-Trained"

reconstructions_df = pd.read_csv("reconstructions_per_slice.csv")
reconstructions_df.loc[reconstructions_df["Method"] == "CIRIM", "Method"] = "Pre-Trained"

# if MTLRS_ft in each fold then replace MTLRS with MTLRS_ft and remove MTLRS_ft
for fold in range(6):
    if "MTLRS_ft" in segmentations_df[segmentations_df["Fold"] == fold]["Method"].unique():
        segmentations_df = segmentations_df[
            ~((segmentations_df["Fold"] == fold) & (segmentations_df["Method"] == "MTLRS"))
        ]
        segmentations_df.loc[
            (segmentations_df["Fold"] == fold) & (segmentations_df["Method"] == "MTLRS_ft"), "Method"
        ] = "MTLRS"
    if "MTLRS_ft" in reconstructions_df[reconstructions_df["Fold"] == fold]["Method"].unique():
        reconstructions_df.loc[
            (reconstructions_df["Fold"] == fold) & (reconstructions_df["Method"] == "MTLRS_ft"), "Method"
        ] = "MTLRS"
        reconstructions_df = reconstructions_df[
            ~((reconstructions_df["Fold"] == fold) & (reconstructions_df["Method"] == "MTLRS_ft"))
        ]

segmentations_df = segmentations_df[~segmentations_df["Method"].isin(["RECSEGNET", "IDSLR", "SEGNET", "SERANET"])]
reconstructions_df = reconstructions_df[
    ~reconstructions_df["Method"].isin(["RECSEGNET", "IDSLR", "SEGNET", "SERANET", "End-to-End"])
]

# remove fold 3
segmentations_df = segmentations_df[segmentations_df["Fold"] != 3]
reconstructions_df = reconstructions_df[reconstructions_df["Fold"] != 3]
# subtract 1 to folds 5-6
# segmentations_df.loc[segmentations_df["Fold"] > 3, "Fold"] = segmentations_df.loc[segmentations_df["Fold"] > 3, "Fold"] - 1
# reconstructions_df.loc[reconstructions_df["Fold"] > 3, "Fold"] = reconstructions_df.loc[reconstructions_df["Fold"] > 3, "Fold"] - 1

# average over folds
reconstructions_df = reconstructions_df.groupby(["Subject", "Method"]).mean().reset_index()
reconstructions_df["Subject"] = reconstructions_df["Subject"].str.split("_").str[0]
reconstructions_df = reconstructions_df.groupby(["Subject", "Method"]).mean().reset_index()

segmentations_df = segmentations_df.groupby(["Subject", "Method"]).mean().reset_index()
segmentations_df["Subject"] = segmentations_df["Subject"].str.split("_").str[0]
segmentations_df = segmentations_df.groupby(["Subject", "Method"]).mean().reset_index()

# create colormap for each method in segmentations_df
cmap = {
    "MTLRS": "tab:blue",
    "Pre-Trained": "tab:orange",
    "Sequential": "tab:green",
    "End-to-End": "tab:red",
    "Joint": "tab:purple",
}
fontsize = 20
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
# SSIM
sns.boxplot(
    x="Method", y="SSIM", data=reconstructions_df, ax=axes[0, 0], palette=cmap, order=["MTLRS", "Joint", "Pre-Trained"]
)
axes[0, 0].set_ylabel("SSIM")
axes[0, 0].set_xlabel("")
axes[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.04))
# set fontsize
# PSNR
sns.boxplot(
    x="Method", y="PSNR", data=reconstructions_df, ax=axes[0, 1], palette=cmap, order=["MTLRS", "Joint", "Pre-Trained"]
)
axes[0, 1].set_ylabel("PSNR")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylim(28, 36.5)
axes[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(2))
# Dice
sns.boxplot(
    x="Method",
    y="Dice",
    data=segmentations_df,
    ax=axes[1, 0],
    palette=cmap,
    order=["MTLRS", "Joint", "Sequential", "Pre-Trained", "End-to-End"],
)
axes[1, 0].set_ylabel("DICE")
axes[1, 0].set_xlabel("")
# set y axis range
axes[1, 0].set_ylim(0.0, 0.8)
axes[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
# Dice_Lesions
sns.boxplot(
    x="Method",
    y="Dice_Lesions",
    data=segmentations_df,
    ax=axes[1, 1],
    palette=cmap,
    order=["MTLRS", "Joint", "Sequential", "Pre-Trained", "End-to-End"],
)
axes[1, 1].set_ylabel("DICE Lesions")
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylim(0.0, 0.8)
axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(ax.get_xlabel(), fontsize=fontsize)
    plt.ylabel(ax.get_ylabel(), fontsize=fontsize)
plt.tight_layout()
plt.show()
