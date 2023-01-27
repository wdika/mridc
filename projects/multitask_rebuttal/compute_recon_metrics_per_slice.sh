python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction/CIRIM/default/2023-02-14_18-34-27/reconstructions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv CIRIM 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/IDSLR/default/2023-02-09_13-28-16/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv IDSLR 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-12_02-30-41/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss_finetune/default/2023-02-13_14-18-06/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS_ft 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/RECSEGNET/default/2023-02-11_14-45-16/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv RECSEGNET 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/SEGNET/default/2023-02-11_14-23-07/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SEGNET 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/SERANET/default/2023-02-10_20-15-41/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SERANET 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/JRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-13_08-33-42/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv Joint 0
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_0_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold0_TestSet/Reconstruction_Segmentation/E2ECIRIMAttentionUNet/default/2023-02-14_08-46-06/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv End-to-End 0

python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction/CIRIM/default/2023-02-14_18-46-06/reconstructions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv CIRIM 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/IDSLR/default/2023-02-09_13-36-55/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv IDSLR 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-12_22-52-04/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss_finetune/default/2023-02-13_13-18-54/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS_ft 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/RECSEGNET/default/2023-02-11_14-46-25/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv RECSEGNET 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/SEGNET/default/2023-02-11_14-34-53/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SEGNET 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/SERANET/default/2023-02-10_20-17-54/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SERANET 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/JRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-13_13-08-16/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv Joint 1
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_1_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold1_TestSet/Reconstruction_Segmentation/E2ECIRIMAttentionUNet/default/2023-02-13_18-27-56/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv End-to-End 1

python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction/CIRIM/default/2023-02-14_13-54-28/reconstructions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv CIRIM 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/IDSLR/default/2023-02-09_15-58-14/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv IDSLR 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-12_02-39-28/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss_finetune/default/2023-02-13_14-25-21/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS_ft 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/RECSEGNET/default/2023-02-11_14-46-50/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv RECSEGNET 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/SEGNET/default/2023-02-11_14-41-07/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SEGNET 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/SERANET/default/2023-02-10_20-22-38/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SERANET 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/E2ECIRIMAttentionUNet/default/2023-02-14_13-41-26/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv End-to-End 2
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_2_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold2_TestSet/Reconstruction_Segmentation/JRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-13_12-20-17/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv Joint 2

python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction/CIRIM/default/2023-02-13_23-53-18/reconstructions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv CIRIM 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/IDSLR/default/2023-02-09_16-06-15/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv IDSLR 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-12_22-56-39/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss_finetune/default/2023-02-14_08-59-36/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS_ft 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/RECSEGNET/default/2023-02-11_14-48-25/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv RECSEGNET 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/SEGNET/default/2023-02-11_14-44-43/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SEGNET 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/SERANET/default/2023-02-10_20-23-21/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SERANET 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/E2ECIRIMAttentionUNet/default/2023-02-14_13-48-41/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv End-to-End 3
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_3_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold3_TestSet/Reconstruction_Segmentation/JRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-14_08-47-58/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv Joint 3

python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction/CIRIM/default/2023-02-14_18-17-54/reconstructions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv CIRIM 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/IDSLR/default/2023-02-09_16-13-57/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv IDSLR 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-11_20-13-47/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss_finetune/default/2023-02-14_09-00-25/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS_ft 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/RECSEGNET/default/2023-02-11_14-49-42/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv RECSEGNET 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/SEGNET/default/2023-02-10_23-25-59/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SEGNET 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/SERANET/default/2023-02-10_20-29-28/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SERANET 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/E2ECIRIMAttentionUNet/default/2023-02-14_03-04-18/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv End-to-End 4
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/folds/fold_4_test.json \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold4_TestSet/Reconstruction_Segmentation/JRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-14_08-49-34/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv Joint 4

python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction/CIRIM/default/2022-11-17_01-21-31/reconstructions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv CIRIM 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction_Segmentation/IDSLR/default/2022-11-17_10-51-31/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv IDSLR 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_fold5_TestSet/Reconstruction_Segmentation/MTL_MRIRS_0.1_reconstruction_loss_0.9_segmentation_loss/default/2023-02-14_13-00-08/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv MTLRS 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction_Segmentation/RECSEGNET/default/2022-11-17_10-52-07/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv RECSEGNET 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction_Segmentation/SEGNET/default/2022-11-17_10-52-12/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SEGNET 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction_Segmentation/SERANET/default/2022-11-17_10-52-15/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv SERANET 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction_Segmentation/SERANET/default/2022-11-17_10-52-15/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv End-to-End 5
python projects/multitask_rebuttal/compute_reconstruction_metrics_per_slice.py /data/projects/tecfidera/data/synthesized/data/multicoil_test \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning/predictions/TECFIDERA_3D_FLAIR_Synthesized_Poisson2D_8x_TestSet/Reconstruction_Segmentation/JRS_CIRIMAttentionUNet_0.1_reconstruction_loss_0.9_segmentation_loss/default/2022-12-10_18-26-35/predictions \
    /data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/evaluations/reconstructions_per_slice.csv Joint 5
