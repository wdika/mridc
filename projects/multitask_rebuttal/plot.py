# coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

segmentations_df = pd.read_csv(
    "/data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/segmentations.csv"
)
segmentations_df.loc[segmentations_df["Method"] == "AttentionUNet", "Method"] = "Pre-Trained"

reconstructions_df = pd.read_csv(
    "/data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions.csv"
)
reconstructions_df.loc[reconstructions_df["Method"] == "CIRIM", "Method"] = "Pre-Trained"

for fold in range(5):
    if "MTLRS_ft" in segmentations_df[segmentations_df["Fold"] == fold]["Method"].values:
        segmentations_df = segmentations_df[
            (segmentations_df["Fold"] != fold) | (segmentations_df["Method"] != "MTLRS")
        ]
        segmentations_df.loc[
            (segmentations_df["Fold"] == fold) & (segmentations_df["Method"] == "MTLRS_ft"), "Method"
        ] = "MTLRS"
    if "MTLRS_ft" in reconstructions_df[reconstructions_df["Fold"] == fold]["Method"].values:
        reconstructions_df = reconstructions_df[
            (reconstructions_df["Fold"] != fold) | (reconstructions_df["Method"] != "MTLRS")
        ]
        reconstructions_df.loc[
            (reconstructions_df["Fold"] == fold) & (reconstructions_df["Method"] == "MTLRS_ft"), "Method"
        ] = "MTLRS"

segmentations_df = segmentations_df[~segmentations_df["Method"].isin(["RECSEGNET", "IDSLR", "SEGNET", "SERANET"])]
reconstructions_df = reconstructions_df[
    ~reconstructions_df["Method"].isin(["RECSEGNET", "IDSLR", "SEGNET", "SERANET", "End-to-End"])
]
segmentations_df.loc[segmentations_df["Method"] == "JRS", "Method"] = "Joint"
reconstructions_df.loc[reconstructions_df["Method"] == "JRS", "Method"] = "Joint"

# remove fold 5
segmentations_df = segmentations_df[segmentations_df["Fold"] != 5]
reconstructions_df = reconstructions_df[reconstructions_df["Fold"] != 5]

segmentations_df["Subject"] = segmentations_df["Subject"].str.split("_").str[0]
reconstructions_df["Subject"] = reconstructions_df["Subject"].str.split("_").str[0]

segmentations_df = segmentations_df.groupby(["Subject", "Method"]).mean().reset_index()
reconstructions_df = reconstructions_df.groupby(["Subject", "Method"]).mean().reset_index()

# create colormap for each method in segmentations_df
cmap = {
    "MTLRS": "tab:blue",
    "Pre-Trained": "tab:orange",
    "Sequential": "tab:green",
    "End-to-End": "tab:red",
    "Joint": "tab:purple",
}
fontsize = 20
# sns.set(style="whitegrid")
# fig, ax = plt.subplots(2, 2, figsize=(20, 10))
# sns.boxplot(x="Method", y="SSIM", data=reconstructions_df.sort_values(by=['SSIM'], ascending=False), ax=ax[0, 0], palette=cmap)
# ax[0, 0].set_xlabel('')
# ax[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(.1))
# ax[0, 0].set_ylim(0.8, 1)
# sns.boxplot(x="Method", y="PSNR", data=reconstructions_df.sort_values(by=['PSNR'], ascending=False), ax=ax[0, 1], palette=cmap)
# ax[0, 1].set_xlabel('')
# ax[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(1))
# ax[0, 1].set_ylim(25, 39)
# sns.boxplot(x="Method", y="Dice", data=segmentations_df.sort_values(by=['Dice'], ascending=False), ax=ax[1, 0], palette=cmap)
# ax[1, 0].set_xlabel('')
# ax[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(.1))
# # ax[1, 0].set_ylim(0.2, 0.8)
# sns.boxplot(x="Method", y="Dice_Lesions", data=segmentations_df.sort_values(by=['Dice_Lesions'], ascending=False), ax=ax[1, 1], palette=cmap)
# ax[1, 1].set_xlabel('')
# ax[1, 1].set_ylabel('DICE Lesions')
# ax[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(.1))
# # ax[1, 1].set_ylim(0.2, 0.8)
# for ax in fig.axes:
#     plt.sca(ax)
#     plt.xticks(fontsize=fontsize)
#     plt.yticks(fontsize=fontsize)
#     plt.xlabel(ax.get_xlabel(), fontsize=fontsize)
#     plt.ylabel(ax.get_ylabel(), fontsize=fontsize)
# plt.tight_layout()
# plt.show()

# sns.set_theme(style="whitegrid")
# sns.set_context("paper", font_scale=1.5)
# # custom palette, "MTLRS"==blue, "Pre-Trained"==orange, "Joint"==purple, "End-to-End"==red, "Sequential"==green
# # palette = ["#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#2ca02c"]
# # # assign colors to methods
# # segmentations_df["Method"] = pd.Categorical(segmentations_df["Method"], categories=["MTLRS", "Pre-Trained", "Joint", "End-to-End", "Sequential"], ordered=True)
# # reconstructions_df["Method"] = pd.Categorical(reconstructions_df["Method"], categories=["MTLRS", "Pre-Trained", "Joint"], ordered=True)
#
# subplot DICE and DICE Lesions for folds 0-4
fig, axes = plt.subplots(4, 5, figsize=(20, 10))
for fold in range(5):
    # SSIM
    sns.boxplot(
        x="Method",
        y="SSIM",
        data=reconstructions_df.sort_values(by=["SSIM"], ascending=False),
        ax=axes[0, fold],
        palette=cmap,
    )
    axes[0, fold].set_title("Fold {}".format(fold + 1))
    axes[0, fold].set_xlabel("")
    if fold == 0:
        axes[0, fold].set_ylabel("SSIM")
        axes[0, fold].set_ylim(0.0, 1.0)
    else:
        axes[0, fold].set_ylabel("")
        axes[0, fold].set_yticklabels([])
    # PSNR
    sns.boxplot(
        x="Method",
        y="PSNR",
        data=reconstructions_df.sort_values(by=["PSNR"], ascending=False),
        ax=axes[1, fold],
        palette=cmap,
    )
    axes[1, fold].set_xlabel("")
    if fold == 0:
        axes[1, fold].set_ylabel("PSNR")
        axes[1, fold].set_ylim(0.0, 50.0)
    else:
        axes[1, fold].set_ylabel("")
        axes[1, fold].set_yticklabels([])
    # DICE
    sns.boxplot(
        x="Method",
        y="Dice",
        data=segmentations_df[segmentations_df["Fold"] == fold].sort_values(by=["Dice"], ascending=False),
        ax=axes[2, fold],
        palette=cmap,
    )
    axes[2, fold].set_xlabel("")
    if fold == 0:
        axes[2, fold].set_ylabel("Dice")
        axes[2, fold].set_ylim(0.35, 0.9)
    else:
        axes[2, fold].set_ylabel("")
        axes[2, fold].set_yticklabels([])
    # DICE Lesions
    sns.boxplot(
        x="Method",
        y="Dice_Lesions",
        data=segmentations_df[segmentations_df["Fold"] == fold].sort_values(by=["Dice_Lesions"], ascending=False),
        ax=axes[3, fold],
        palette=cmap,
    )
    axes[3, fold].set_xlabel("")
    if fold == 0:
        axes[3, fold].set_ylabel("Dice Lesions")
        axes[3, fold].set_ylim(0.0, 0.9)
    else:
        axes[3, fold].set_ylabel("")
        axes[3, fold].set_yticklabels([])
for ax in axes.flat:
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(8)
plt.tight_layout()
plt.show()

# rename fold 1-4 to fold 0
# segmentations_df.loc[segmentations_df["Fold"] != 0, "Fold"] = 0
# reconstructions_df.loc[reconstructions_df["Fold"] != 0, "Fold"] = 0
#
# segmentations_df = segmentations_df.sort_values(by=["Dice"], ascending=False)
# reconstructions_df = reconstructions_df.sort_values(by=["SSIM"], ascending=False)
#
# sns.set_theme(style="whitegrid")
# sns.set_context("paper", font_scale=1.5)
# palette = ["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c", "#d62728"]
# segmentations_df["Method"] = pd.Categorical(segmentations_df["Method"], categories=["MTLRS", "Joint", "Pre-Trained", "Sequential", "End-to-End"], ordered=False)
# reconstructions_df["Method"] = pd.Categorical(reconstructions_df["Method"], categories=["MTLRS", "Joint", "Pre-Trained"], ordered=False)
#
# # plot average of five-folds for each method
# fig, axes = plt.subplots(2, 2, figsize=(10, 5))
# # SSIM
# sns.boxplot(x="Method", y="SSIM", data=reconstructions_df[(reconstructions_df["Method"] != "End-to-End") & (reconstructions_df["Method"] != "Sequential")], ax=axes[0, 0], palette=palette)
# axes[0, 0].set_xlabel("")
# axes[0, 0].set_ylabel("SSIM")
# axes[0, 0].set_ylim(0.8, 1.0)
# # PSNR
# sns.boxplot(x="Method", y="PSNR", data=reconstructions_df[(reconstructions_df["Method"] != "End-to-End") & (reconstructions_df["Method"] != "Sequential")], ax=axes[0, 1], palette=palette)
# axes[0, 1].set_xlabel("")
# axes[0, 1].set_ylabel("PSNR")
# axes[0, 1].set_ylim(25, 40.0)
# # DICE
# sns.boxplot(x="Method", y="Dice", data=segmentations_df, ax=axes[1, 0], palette=palette)
# axes[1, 0].set_xlabel("")
# axes[1, 0].set_ylabel("Dice")
# axes[1, 0].set_ylim(0.4, 0.9)
# # DICE Lesions
# sns.boxplot(x="Method", y="Dice_Lesions", data=segmentations_df, ax=axes[1, 1], palette=palette)
# axes[1, 1].set_xlabel("")
# axes[1, 1].set_ylabel("Dice Lesions")
# axes[1, 1].set_ylim(0.0, 0.8)
# plt.tight_layout()
# plt.show()
