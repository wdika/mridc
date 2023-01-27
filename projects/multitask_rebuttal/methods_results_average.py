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

segmentations_df = segmentations_df[
    ~segmentations_df["Method"].isin(["Pre-Trained", "Joint", "Sequential", "End-to-End"])
]
reconstructions_df = reconstructions_df[
    ~reconstructions_df["Method"].isin(["Pre-Trained", "Joint", "Sequential", "End-to-End"])
]

# remove fold 3
segmentations_df = segmentations_df[segmentations_df["Fold"] != 3]
reconstructions_df = reconstructions_df[reconstructions_df["Fold"] != 3]
# subtract 1 to folds 4-6
segmentations_df.loc[segmentations_df["Fold"] > 2, "Fold"] = (
    segmentations_df.loc[segmentations_df["Fold"] > 2, "Fold"] - 1
)
reconstructions_df.loc[reconstructions_df["Fold"] > 2, "Fold"] = (
    reconstructions_df.loc[reconstructions_df["Fold"] > 2, "Fold"] - 1
)

# compute mean and std for each fold and method and each metric
segmentations_df = (
    segmentations_df.groupby(["Method"]).agg({"Dice": ["mean", "std"], "Dice_Lesions": ["mean", "std"]}).reset_index()
)
segmentations_df["Dice"] = segmentations_df["Dice"].apply(lambda x: f"{x['mean']:.3f} $\pm$ {x['std']:.3f}", axis=1)
segmentations_df["Dice_Lesions"] = segmentations_df["Dice_Lesions"].apply(
    lambda x: f"{x['mean']:.3f} $\pm$ {x['std']:.3f}", axis=1
)
# remove std from the table
segmentations_df.columns = segmentations_df.columns.droplevel(1)
# remove duplicate columns
segmentations_df = segmentations_df.loc[:, ~segmentations_df.columns.duplicated()]
# sort by Dice
segmentations_df = segmentations_df.sort_values(by=["Dice"], ascending=[False]).reset_index()
# remove index column
segmentations_df = segmentations_df.drop(columns=["index"])

reconstructions_df = (
    reconstructions_df.groupby(["Method"]).agg({"SSIM": ["mean", "std"], "PSNR": ["mean", "std"]}).reset_index()
)
reconstructions_df["SSIM"] = reconstructions_df["SSIM"].apply(
    lambda x: f"{x['mean']:.3f} $\pm$ {x['std']:.3f}", axis=1
)
reconstructions_df["PSNR"] = reconstructions_df["PSNR"].apply(
    lambda x: f"{x['mean']:.2f} $\pm$ {x['std']:.2f}", axis=1
)
# remove std from the table
reconstructions_df.columns = reconstructions_df.columns.droplevel(1)
# remove duplicate columns
reconstructions_df = reconstructions_df.loc[:, ~reconstructions_df.columns.duplicated()]
# sort by SSIM
reconstructions_df = reconstructions_df.sort_values(by=["SSIM"], ascending=[False]).reset_index()
# remove index column
reconstructions_df = reconstructions_df.drop(columns=["index"])

# merge segmentations and reconstructions, if Method not in reconstructions_df then add it with NaN
merged_df = segmentations_df.merge(reconstructions_df, on=["Method"], how="outer")
# fill NaN with empty string
merged_df = merged_df.fillna("")
# SSIM on column 2, PSNR on column 3, Dice on column 4, Dice_Lesions on column 5
merged_df = merged_df[["Method", "SSIM", "PSNR", "Dice", "Dice_Lesions"]]
# if SSIM is equal between two methods then sort by PSNR
merged_df = merged_df.sort_values(
    by=["SSIM", "PSNR", "Dice", "Dice_Lesions"], ascending=[False, False, False, False]
).reset_index()
# remove index column
merged_df = merged_df.drop(columns=["index"])

# print them to latex with midrule
print(merged_df.style.to_latex())
print("\\midrule")