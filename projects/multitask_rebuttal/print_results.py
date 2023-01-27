# coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

segmentation_df = pd.read_csv(
    "/data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/segmentations.csv"
)

# remove AttentionUNet
segmentation_df = segmentation_df[segmentation_df["Method"] != "AttentionUNet"]

# split folds to separate dataframes
segmentation_df_fold_0 = segmentation_df[segmentation_df["Fold"] == 0]
segmentation_df_fold_1 = segmentation_df[segmentation_df["Fold"] == 1]
segmentation_df_fold_2 = segmentation_df[segmentation_df["Fold"] == 2]
segmentation_df_fold_3 = segmentation_df[segmentation_df["Fold"] == 3]
segmentation_df_fold_4 = segmentation_df[segmentation_df["Fold"] == 4]

segmentation_df_fold_0 = segmentation_df_fold_0.groupby(["Fold", "Method"]).agg(
    {"Dice": ["mean", "std"], "Dice_Lesions": ["mean", "std"]}
)
segmentation_df_fold_0.columns = ["Dice_Mean", "Dice_Std", "Dice_Lesions_Mean", "Dice_Lesions_Std"]
# round and concatenate mean and std
segmentation_df_fold_0["DICE"] = (
    segmentation_df_fold_0["Dice_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_0["Dice_Std"].apply(lambda x: f"{x:.3f}")
)

segmentation_df_fold_0 = segmentation_df_fold_0.drop(columns=["Dice_Mean", "Dice_Std"])
segmentation_df_fold_0["DICE Lesions"] = (
    segmentation_df_fold_0["Dice_Lesions_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_0["Dice_Lesions_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_0 = segmentation_df_fold_0.drop(columns=["Dice_Lesions_Mean", "Dice_Lesions_Std"])
segmentation_df_fold_0 = segmentation_df_fold_0.sort_values(by=["DICE"], ascending=False)
segmentation_df_fold_0 = segmentation_df_fold_0.reset_index()
# drop fold column
segmentation_df_fold_0 = segmentation_df_fold_0.drop(columns=["Fold"])

segmentation_df_fold_1 = segmentation_df_fold_1.groupby(["Fold", "Method"]).agg(
    {"Dice": ["mean", "std"], "Dice_Lesions": ["mean", "std"]}
)
segmentation_df_fold_1.columns = ["Dice_Mean", "Dice_Std", "Dice_Lesions_Mean", "Dice_Lesions_Std"]
# round and concatenate mean and std
segmentation_df_fold_1["DICE"] = (
    segmentation_df_fold_1["Dice_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_1["Dice_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_1 = segmentation_df_fold_1.drop(columns=["Dice_Mean", "Dice_Std"])
segmentation_df_fold_1["DICE Lesions"] = (
    segmentation_df_fold_1["Dice_Lesions_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_1["Dice_Lesions_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_1 = segmentation_df_fold_1.drop(columns=["Dice_Lesions_Mean", "Dice_Lesions_Std"])
segmentation_df_fold_1 = segmentation_df_fold_1.sort_values(by=["DICE"], ascending=False)
segmentation_df_fold_1 = segmentation_df_fold_1.reset_index()
# drop fold column
segmentation_df_fold_1 = segmentation_df_fold_1.drop(columns=["Fold"])

segmentation_df_fold_2 = segmentation_df_fold_2.groupby(["Fold", "Method"]).agg(
    {"Dice": ["mean", "std"], "Dice_Lesions": ["mean", "std"]}
)
segmentation_df_fold_2.columns = ["Dice_Mean", "Dice_Std", "Dice_Lesions_Mean", "Dice_Lesions_Std"]
# round and concatenate mean and std
segmentation_df_fold_2["DICE"] = (
    segmentation_df_fold_2["Dice_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_2["Dice_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_2 = segmentation_df_fold_2.drop(columns=["Dice_Mean", "Dice_Std"])
segmentation_df_fold_2["DICE Lesions"] = (
    segmentation_df_fold_2["Dice_Lesions_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_2["Dice_Lesions_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_2 = segmentation_df_fold_2.drop(columns=["Dice_Lesions_Mean", "Dice_Lesions_Std"])
segmentation_df_fold_2 = segmentation_df_fold_2.sort_values(by=["DICE"], ascending=False)
segmentation_df_fold_2 = segmentation_df_fold_2.reset_index()
# drop fold column
segmentation_df_fold_2 = segmentation_df_fold_2.drop(columns=["Fold"])

segmentation_df_fold_3 = segmentation_df_fold_3.groupby(["Fold", "Method"]).agg(
    {"Dice": ["mean", "std"], "Dice_Lesions": ["mean", "std"]}
)
segmentation_df_fold_3.columns = ["Dice_Mean", "Dice_Std", "Dice_Lesions_Mean", "Dice_Lesions_Std"]
# round and concatenate mean and std
segmentation_df_fold_3["DICE"] = (
    segmentation_df_fold_3["Dice_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_3["Dice_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_3 = segmentation_df_fold_3.drop(columns=["Dice_Mean", "Dice_Std"])
segmentation_df_fold_3["DICE Lesions"] = (
    segmentation_df_fold_3["Dice_Lesions_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_3["Dice_Lesions_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_3 = segmentation_df_fold_3.drop(columns=["Dice_Lesions_Mean", "Dice_Lesions_Std"])
segmentation_df_fold_3 = segmentation_df_fold_3.sort_values(by=["DICE"], ascending=False)
segmentation_df_fold_3 = segmentation_df_fold_3.reset_index()
# drop fold column
segmentation_df_fold_3 = segmentation_df_fold_3.drop(columns=["Fold"])

segmentation_df_fold_4 = segmentation_df_fold_4.groupby(["Fold", "Method"]).agg(
    {"Dice": ["mean", "std"], "Dice_Lesions": ["mean", "std"]}
)
segmentation_df_fold_4.columns = ["Dice_Mean", "Dice_Std", "Dice_Lesions_Mean", "Dice_Lesions_Std"]
# round and concatenate mean and std
segmentation_df_fold_4["DICE"] = (
    segmentation_df_fold_4["Dice_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_4["Dice_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_4 = segmentation_df_fold_4.drop(columns=["Dice_Mean", "Dice_Std"])
segmentation_df_fold_4["DICE Lesions"] = (
    segmentation_df_fold_4["Dice_Lesions_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + segmentation_df_fold_4["Dice_Lesions_Std"].apply(lambda x: f"{x:.3f}")
)
segmentation_df_fold_4 = segmentation_df_fold_4.drop(columns=["Dice_Lesions_Mean", "Dice_Lesions_Std"])
segmentation_df_fold_4 = segmentation_df_fold_4.sort_values(by=["DICE"], ascending=False)
segmentation_df_fold_4 = segmentation_df_fold_4.reset_index()
# drop fold column
segmentation_df_fold_4 = segmentation_df_fold_4.drop(columns=["Fold"])


reconstruction_df = pd.read_csv(
    "/data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions.csv"
)

reconstruction_df_fold_0 = reconstruction_df[reconstruction_df["Fold"] == 0]
reconstruction_df_fold_1 = reconstruction_df[reconstruction_df["Fold"] == 1]
reconstruction_df_fold_2 = reconstruction_df[reconstruction_df["Fold"] == 2]
reconstruction_df_fold_3 = reconstruction_df[reconstruction_df["Fold"] == 3]
reconstruction_df_fold_4 = reconstruction_df[reconstruction_df["Fold"] == 4]

reconstruction_df_fold_0 = reconstruction_df_fold_0.groupby(["Fold", "Method"]).agg(
    {"SSIM": ["mean", "std"], "PSNR": ["mean", "std"]}
)
reconstruction_df_fold_0.columns = ["SSIM_Mean", "SSIM_Std", "PSNR_Mean", "PSNR_Std"]
# round and concatenate mean and std
reconstruction_df_fold_0["SSIM"] = (
    reconstruction_df_fold_0["SSIM_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + reconstruction_df_fold_0["SSIM_Std"].apply(lambda x: f"{x:.3f}")
)
reconstruction_df_fold_0 = reconstruction_df_fold_0.drop(columns=["SSIM_Mean", "SSIM_Std"])
reconstruction_df_fold_0["PSNR"] = (
    reconstruction_df_fold_0["PSNR_Mean"].apply(lambda x: f"{x:.2f}")
    + " $\pm$ "
    + reconstruction_df_fold_0["PSNR_Std"].apply(lambda x: f"{x:.2f}")
)
reconstruction_df_fold_0 = reconstruction_df_fold_0.drop(columns=["PSNR_Mean", "PSNR_Std"])
reconstruction_df_fold_0 = reconstruction_df_fold_0.sort_values(by=["SSIM"], ascending=False)
reconstruction_df_fold_0 = reconstruction_df_fold_0.reset_index()
# drop fold column
reconstruction_df_fold_0 = reconstruction_df_fold_0.drop(columns=["Fold"])

reconstruction_df_fold_1 = reconstruction_df_fold_1.groupby(["Fold", "Method"]).agg(
    {"SSIM": ["mean", "std"], "PSNR": ["mean", "std"]}
)
reconstruction_df_fold_1.columns = ["SSIM_Mean", "SSIM_Std", "PSNR_Mean", "PSNR_Std"]
# round and concatenate mean and std
reconstruction_df_fold_1["SSIM"] = (
    reconstruction_df_fold_1["SSIM_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + reconstruction_df_fold_1["SSIM_Std"].apply(lambda x: f"{x:.3f}")
)
reconstruction_df_fold_1 = reconstruction_df_fold_1.drop(columns=["SSIM_Mean", "SSIM_Std"])
reconstruction_df_fold_1["PSNR"] = (
    reconstruction_df_fold_1["PSNR_Mean"].apply(lambda x: f"{x:.2f}")
    + " $\pm$ "
    + reconstruction_df_fold_1["PSNR_Std"].apply(lambda x: f"{x:.2f}")
)
reconstruction_df_fold_1 = reconstruction_df_fold_1.drop(columns=["PSNR_Mean", "PSNR_Std"])
reconstruction_df_fold_1 = reconstruction_df_fold_1.sort_values(by=["SSIM"], ascending=False)
reconstruction_df_fold_1 = reconstruction_df_fold_1.reset_index()
# drop fold column
reconstruction_df_fold_1 = reconstruction_df_fold_1.drop(columns=["Fold"])

reconstruction_df_fold_2 = reconstruction_df_fold_2.groupby(["Fold", "Method"]).agg(
    {"SSIM": ["mean", "std"], "PSNR": ["mean", "std"]}
)
reconstruction_df_fold_2.columns = ["SSIM_Mean", "SSIM_Std", "PSNR_Mean", "PSNR_Std"]
# round and concatenate mean and std
reconstruction_df_fold_2["SSIM"] = (
    reconstruction_df_fold_2["SSIM_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + reconstruction_df_fold_2["SSIM_Std"].apply(lambda x: f"{x:.3f}")
)
reconstruction_df_fold_2 = reconstruction_df_fold_2.drop(columns=["SSIM_Mean", "SSIM_Std"])
reconstruction_df_fold_2["PSNR"] = (
    reconstruction_df_fold_2["PSNR_Mean"].apply(lambda x: f"{x:.2f}")
    + " $\pm$ "
    + reconstruction_df_fold_2["PSNR_Std"].apply(lambda x: f"{x:.2f}")
)
reconstruction_df_fold_2 = reconstruction_df_fold_2.drop(columns=["PSNR_Mean", "PSNR_Std"])
reconstruction_df_fold_2 = reconstruction_df_fold_2.sort_values(by=["SSIM"], ascending=False)
reconstruction_df_fold_2 = reconstruction_df_fold_2.reset_index()
# drop fold column
reconstruction_df_fold_2 = reconstruction_df_fold_2.drop(columns=["Fold"])

reconstruction_df_fold_3 = reconstruction_df_fold_3.groupby(["Fold", "Method"]).agg(
    {"SSIM": ["mean", "std"], "PSNR": ["mean", "std"]}
)
reconstruction_df_fold_3.columns = ["SSIM_Mean", "SSIM_Std", "PSNR_Mean", "PSNR_Std"]
# round and concatenate mean and std
reconstruction_df_fold_3["SSIM"] = (
    reconstruction_df_fold_3["SSIM_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + reconstruction_df_fold_3["SSIM_Std"].apply(lambda x: f"{x:.3f}")
)
reconstruction_df_fold_3 = reconstruction_df_fold_3.drop(columns=["SSIM_Mean", "SSIM_Std"])
reconstruction_df_fold_3["PSNR"] = (
    reconstruction_df_fold_3["PSNR_Mean"].apply(lambda x: f"{x:.2f}")
    + " $\pm$ "
    + reconstruction_df_fold_3["PSNR_Std"].apply(lambda x: f"{x:.2f}")
)
reconstruction_df_fold_3 = reconstruction_df_fold_3.drop(columns=["PSNR_Mean", "PSNR_Std"])
reconstruction_df_fold_3 = reconstruction_df_fold_3.sort_values(by=["SSIM"], ascending=False)
reconstruction_df_fold_3 = reconstruction_df_fold_3.reset_index()
# drop fold column
reconstruction_df_fold_3 = reconstruction_df_fold_3.drop(columns=["Fold"])

reconstruction_df_fold_4 = reconstruction_df_fold_4.groupby(["Fold", "Method"]).agg(
    {"SSIM": ["mean", "std"], "PSNR": ["mean", "std"]}
)
reconstruction_df_fold_4.columns = ["SSIM_Mean", "SSIM_Std", "PSNR_Mean", "PSNR_Std"]
# round and concatenate mean and std
reconstruction_df_fold_4["SSIM"] = (
    reconstruction_df_fold_4["SSIM_Mean"].apply(lambda x: f"{x:.3f}")
    + " $\pm$ "
    + reconstruction_df_fold_4["SSIM_Std"].apply(lambda x: f"{x:.3f}")
)
reconstruction_df_fold_4 = reconstruction_df_fold_4.drop(columns=["SSIM_Mean", "SSIM_Std"])
reconstruction_df_fold_4["PSNR"] = (
    reconstruction_df_fold_4["PSNR_Mean"].apply(lambda x: f"{x:.2f}")
    + " $\pm$ "
    + reconstruction_df_fold_4["PSNR_Std"].apply(lambda x: f"{x:.2f}")
)
reconstruction_df_fold_4 = reconstruction_df_fold_4.drop(columns=["PSNR_Mean", "PSNR_Std"])
reconstruction_df_fold_4 = reconstruction_df_fold_4.sort_values(by=["SSIM"], ascending=False)
reconstruction_df_fold_4 = reconstruction_df_fold_4.reset_index()
# drop fold column
reconstruction_df_fold_4 = reconstruction_df_fold_4.drop(columns=["Fold"])

# concatenate reconstruction and segmentation dataframes per Method
reconstruction_segmentation_df_fold_0 = pd.concat([reconstruction_df_fold_0, segmentation_df_fold_0], axis=1)
# remove duplicate column Method
reconstruction_segmentation_df_fold_0 = reconstruction_segmentation_df_fold_0.loc[
    :, ~reconstruction_segmentation_df_fold_0.columns.duplicated()
]
# reset index to Method
reconstruction_segmentation_df_fold_0 = reconstruction_segmentation_df_fold_0.set_index("Method")

reconstruction_segmentation_df_fold_1 = pd.concat([reconstruction_df_fold_1, segmentation_df_fold_1], axis=1)
# remove duplicate column Method
reconstruction_segmentation_df_fold_1 = reconstruction_segmentation_df_fold_1.loc[
    :, ~reconstruction_segmentation_df_fold_1.columns.duplicated()
]
# reset index to Method
reconstruction_segmentation_df_fold_1 = reconstruction_segmentation_df_fold_1.set_index("Method")

reconstruction_segmentation_df_fold_2 = pd.concat([reconstruction_df_fold_2, segmentation_df_fold_2], axis=1)
# remove duplicate column Method
reconstruction_segmentation_df_fold_2 = reconstruction_segmentation_df_fold_2.loc[
    :, ~reconstruction_segmentation_df_fold_2.columns.duplicated()
]
# reset index to Method
reconstruction_segmentation_df_fold_2 = reconstruction_segmentation_df_fold_2.set_index("Method")

reconstruction_segmentation_df_fold_3 = pd.concat([reconstruction_df_fold_3, segmentation_df_fold_3], axis=1)
# remove duplicate column Method
reconstruction_segmentation_df_fold_3 = reconstruction_segmentation_df_fold_3.loc[
    :, ~reconstruction_segmentation_df_fold_3.columns.duplicated()
]
# reset index to Method
reconstruction_segmentation_df_fold_3 = reconstruction_segmentation_df_fold_3.set_index("Method")

reconstruction_segmentation_df_fold_4 = pd.concat([reconstruction_df_fold_4, segmentation_df_fold_4], axis=1)
# remove duplicate column Method
reconstruction_segmentation_df_fold_4 = reconstruction_segmentation_df_fold_4.loc[
    :, ~reconstruction_segmentation_df_fold_4.columns.duplicated()
]
# reset index to Method
reconstruction_segmentation_df_fold_4 = reconstruction_segmentation_df_fold_4.set_index("Method")

# if MTLRS_ft is in the dataframe, remove MTLRS and rename MTLRS_ft to MTLRS
if "MTLRS_ft" in reconstruction_segmentation_df_fold_0.index:
    reconstruction_segmentation_df_fold_0 = reconstruction_segmentation_df_fold_0.drop("MTLRS")
    reconstruction_segmentation_df_fold_0 = reconstruction_segmentation_df_fold_0.rename(index={"MTLRS_ft": "MTLRS"})
if "MTLRS_ft" in reconstruction_segmentation_df_fold_1.index:
    reconstruction_segmentation_df_fold_1 = reconstruction_segmentation_df_fold_1.drop("MTLRS")
    reconstruction_segmentation_df_fold_1 = reconstruction_segmentation_df_fold_1.rename(index={"MTLRS_ft": "MTLRS"})
if "MTLRS_ft" in reconstruction_segmentation_df_fold_2.index:
    reconstruction_segmentation_df_fold_2 = reconstruction_segmentation_df_fold_2.drop("MTLRS")
    reconstruction_segmentation_df_fold_2 = reconstruction_segmentation_df_fold_2.rename(index={"MTLRS_ft": "MTLRS"})
if "MTLRS_ft" in reconstruction_segmentation_df_fold_3.index:
    reconstruction_segmentation_df_fold_3 = reconstruction_segmentation_df_fold_3.drop("MTLRS")
    reconstruction_segmentation_df_fold_3 = reconstruction_segmentation_df_fold_3.rename(index={"MTLRS_ft": "MTLRS"})
if "MTLRS_ft" in reconstruction_segmentation_df_fold_4.index:
    reconstruction_segmentation_df_fold_4 = reconstruction_segmentation_df_fold_4.drop("MTLRS")
    reconstruction_segmentation_df_fold_4 = reconstruction_segmentation_df_fold_4.rename(index={"MTLRS_ft": "MTLRS"})

# accumulate reconstruction and segmentation dataframes per fold0
reconstruction_segmentation_df_mean = reconstruction_segmentation_df_fold_0.copy()
reconstruction_segmentation_df_mean["SSIM_mean"] = (
    reconstruction_segmentation_df_fold_0["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_1["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_2["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_3["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_4["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
)
reconstruction_segmentation_df_mean["SSIM_mean"] = reconstruction_segmentation_df_mean["SSIM_mean"] / 5
reconstruction_segmentation_df_mean["SSIM_std"] = (
    reconstruction_segmentation_df_fold_0["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_1["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_2["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_3["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_4["SSIM"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
)
reconstruction_segmentation_df_mean["SSIM_std"] = reconstruction_segmentation_df_mean["SSIM_std"] / 5
reconstruction_segmentation_df_mean["PSNR_mean"] = (
    reconstruction_segmentation_df_fold_0["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_1["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_2["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_3["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_4["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
)
reconstruction_segmentation_df_mean["PSNR_mean"] = reconstruction_segmentation_df_mean["PSNR_mean"] / 5
reconstruction_segmentation_df_mean["PSNR_std"] = (
    reconstruction_segmentation_df_fold_0["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_1["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_2["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_3["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_4["PSNR"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
)
reconstruction_segmentation_df_mean["PSNR_std"] = reconstruction_segmentation_df_mean["PSNR_std"] / 5
reconstruction_segmentation_df_mean["DICE_mean"] = (
    reconstruction_segmentation_df_fold_0["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_1["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_2["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_3["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_4["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
)
reconstruction_segmentation_df_mean["DICE_mean"] = reconstruction_segmentation_df_mean["DICE_mean"] / 5
reconstruction_segmentation_df_mean["DICE_std"] = (
    reconstruction_segmentation_df_fold_0["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_1["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_2["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_3["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_4["DICE"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
)
reconstruction_segmentation_df_mean["DICE_std"] = reconstruction_segmentation_df_mean["DICE_std"] / 5
reconstruction_segmentation_df_mean["DICE_Lesions_mean"] = (
    reconstruction_segmentation_df_fold_0["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_1["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_2["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_3["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
    + reconstruction_segmentation_df_fold_4["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[0]))
)
reconstruction_segmentation_df_mean["DICE_Lesions_mean"] = reconstruction_segmentation_df_mean["DICE_Lesions_mean"] / 5
reconstruction_segmentation_df_mean["DICE_Lesions_std"] = (
    reconstruction_segmentation_df_fold_0["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_1["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_2["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_3["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
    + reconstruction_segmentation_df_fold_4["DICE Lesions"].apply(lambda x: float(x.split(" $\pm$ ")[1]))
)
reconstruction_segmentation_df_mean["DICE_Lesions_std"] = reconstruction_segmentation_df_mean["DICE_Lesions_std"] / 5

reconstruction_segmentation_df_mean["SSIM"] = (
    reconstruction_segmentation_df_mean["SSIM_mean"].apply(lambda x: str(round(x, 3)))
    + " $\pm$ "
    + reconstruction_segmentation_df_mean["SSIM_std"].apply(lambda x: str(round(x, 3)))
)
reconstruction_segmentation_df_mean["PSNR"] = (
    reconstruction_segmentation_df_mean["PSNR_mean"].apply(lambda x: str(round(x, 2)))
    + " $\pm$ "
    + reconstruction_segmentation_df_mean["PSNR_std"].apply(lambda x: str(round(x, 2)))
)
reconstruction_segmentation_df_mean["DICE"] = (
    reconstruction_segmentation_df_mean["DICE_mean"].apply(lambda x: str(round(x, 3)))
    + " $\pm$ "
    + reconstruction_segmentation_df_mean["DICE_std"].apply(lambda x: str(round(x, 3)))
)
reconstruction_segmentation_df_mean["DICE Lesions"] = (
    reconstruction_segmentation_df_mean["DICE_Lesions_mean"].apply(lambda x: str(round(x, 3)))
    + " $\pm$ "
    + reconstruction_segmentation_df_mean["DICE_Lesions_std"].apply(lambda x: str(round(x, 3)))
)
# remove unnecessary columns
reconstruction_segmentation_df_mean = reconstruction_segmentation_df_mean.drop(
    columns=[
        "SSIM_mean",
        "SSIM_std",
        "PSNR_mean",
        "PSNR_std",
        "DICE_mean",
        "DICE_std",
        "DICE_Lesions_mean",
        "DICE_Lesions_std",
    ]
)

# print(reconstruction_segmentation_df_mean.style.to_latex())
print(reconstruction_segmentation_df_fold_0.style.to_latex())
print(reconstruction_segmentation_df_fold_1.style.to_latex())
print(reconstruction_segmentation_df_fold_2.style.to_latex())
