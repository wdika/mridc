# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
import time
import webbrowser
from pathlib import Path

import streamlit as st
import yaml  # type: ignore


def _isnone_isnull_(x):
    # check if x is None or "None" or null
    if x is None:
        return True
    if isinstance(x, str):
        return x.lower() == "none"
    if isinstance(x, list):
        return x == ["None"]
    return False


st.title("Run an MRI Experiment")

# Select the task
task = st.selectbox("Task", ["Reconstruction", "Segmentation", "Quantitative Imaging", "MultiTask"])

task = task.lower()
if task == "quantitative imaging":
    task = "quantitative"

if task in ["segmentation", "multitask"]:
    with st.expander("Segmentation Configuration", expanded=False):
        complex_data = st.selectbox("Segmentation Complex Data", [True, False])
        segmentation_classes = st.number_input("Segmentation Classes", min_value=1, max_value=999, value=2)
        segmentation_classes_to_remove = st.text_input("Segmentation Classes to Remove", value="[0]")
        segmentation_classes_to_combine = st.text_input("Segmentation Classes to Combine", value="[1, 2]")
        segmentation_classes_to_separate = st.text_input("Segmentation Classes to Separate", value="[3]")
        segmentation_classes_thresholds = st.text_input("Segmentation Classes Thresholds", value="[0.5, 0.5]")

task_dir = os.path.join("mridc/core/app/conf", task)

# Select the mode
mode = st.selectbox("Mode", ["train", "test"])
if mode == "test":
    mode = "run"

pretrained = st.selectbox("Pretrained model", [False, True])
if pretrained or mode == "run":
    checkpoint_path = st.text_input("Checkpoint Path", value="/data/checkpoints")
else:
    checkpoint_path = None

# Select the method
rim_only = False
with st.expander("Method", expanded=True):
    if task == "multitask":
        model_name = st.selectbox(
            "MRI Multitask Model",
            [
                "Image domain Deep Structured Low-Rank network (IDSLR)",
                "Image domain Deep Structured Low-Rank UNet (IDSLRUNet)",
                "Joint MRI Reconstruction and Segmentation (JRS)",
                "Multi-Task Learning for MRI Reconstruction and Segmentation (MTLRS)",
                "Reconstruction Segmentation UNet (RecSegUNet)",
                "Segmentation Network (SegNet)",
                "Segmentation from k-Space with End-to-End Recurrent Attention Network (SERANet)",
            ],
        )
    elif task == "quantitative":
        model_name = st.selectbox(
            "quantitative MRI Model",
            [
                "quantitative Cascades of Independently Recurrent Inference Machines (qCIRIM)",
                "quantitative Recurrent Inference Machines (qRIM)",
                "quantitative Variational Network (qVN)",
            ],
        )
    elif task == "reconstruction":
        model_name = st.selectbox(
            "MRI Reconstruction Model",
            [
                "Cascades of Independently Recurrent Inference Machines (CIRIM)",
                "Convolutional Recurrent Neural Network (CRNN)",
                "Deep Cascade of Convolutional Neural Networks (DCCNN)",
                "Down-Up NET (DUNET)",
                "Feature-level multi-domain network (MultiDomainNet)",
                "Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet)",
                "KIKI-Net: Cross-Domain Convolutional Neural Networks",
                "Learned Primal-Dual Reconstruction (LPDNet)",
                "Recurrent Inference Machines (RIM)",
                "Recurrent Variational Network (RVN)",
                "UNet",
                "Variational Network (VN)",
                "Variable-Splitting Net (VSNet)",
                "XPDNet",
                "Zero-Filled",
            ],
        )
    elif task == "segmentation":
        model_name = st.selectbox(
            "MRI Segmentation Model",
            [
                "Attention UNet (2D)",
                "Dynamic UNet (2D)",
                "Lambda UNet (2D)",
                "Lambda UNet (3D)",
                "UNet (2D)",
                "UNet (3D)",
                "UNetR (2D)",
                "VNet (2D)",
            ],
        )

segmentation_dim = "2D"
with st.expander("Hyperparameters", expanded=False):
    if model_name == "Image domain Deep Structured Low-Rank network (IDSLR)":
        model_name = "IDSLR"
        use_reconstruction_module = st.selectbox("Use Reconstruction Module", [True, False])
        input_channels = st.number_input("Input Channels (Number of coils x 2)", min_value=1, max_value=999, value=64)
        reconstruction_module_output_channels = st.number_input(
            "Reconstruction Module Output Channels (Number of coils x 2)", min_value=1, max_value=999, value=64
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels (Number of classes)", min_value=1, max_value=999, value=2
        )
        channels = st.number_input("Channels", min_value=1, max_value=999, value=64)
        num_pools = st.number_input("Number of Pools", min_value=1, max_value=999, value=4)
        padding = st.selectbox("Padding", [True, False])
        padding_size = st.number_input("Padding Size", min_value=1, max_value=999, value=11)
        drop_prob = st.number_input("Dropout Probability", min_value=0.0, max_value=1.0, value=0.0)
        normalize = st.selectbox("Normalize", [True, False])
        norm_groups = st.number_input("Norm Groups", min_value=1, max_value=999, value=2)
        num_iters = st.number_input("Number of Iterations", min_value=1, max_value=999, value=5)
        total_reconstruction_loss_weight = st.number_input(
            "Total Reconstruction Loss Weight", min_value=0.0, max_value=1.0, value=0.99999
        )
        total_segmentation_loss_weight = st.number_input(
            "Total Segmentation Loss Weight", min_value=0.0, max_value=1.0, value=0.00001
        )
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": use_reconstruction_module,
            "input_channels": input_channels,
            "reconstruction_module_output_channels": reconstruction_module_output_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "channels": channels,
            "num_pools": num_pools,
            "padding": padding,
            "padding_size": padding_size,
            "drop_prob": drop_prob,
            "normalize": normalize,
            "norm_groups": norm_groups,
            "num_iters": num_iters,
            "total_reconstruction_loss_weight": total_reconstruction_loss_weight,
            "total_segmentation_loss_weight": total_segmentation_loss_weight,
            "use_sens_net": use_sens_net,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Image domain Deep Structured Low-Rank UNet (IDSLRUNet)":
        model_name = "IDSLRUNET"
        use_reconstruction_module = st.selectbox("Use Reconstruction Module", [True, False])
        input_channels = st.number_input("Input Channels (Number of coils x 2)", min_value=1, max_value=999, value=64)
        reconstruction_module_output_channels = st.number_input(
            "Reconstruction Module Output Channels (Number of coils x 2)", min_value=1, max_value=999, value=64
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels (Number of classes)", min_value=1, max_value=999, value=2
        )
        channels = st.number_input("Channels", min_value=1, max_value=999, value=64)
        num_pools = st.number_input("Number of Pools", min_value=1, max_value=999, value=4)
        padding = st.selectbox("Padding", [True, False])
        padding_size = st.number_input("Padding Size", min_value=1, max_value=999, value=11)
        drop_prob = st.number_input("Dropout Probability", min_value=0.0, max_value=1.0, value=0.0)
        normalize = st.selectbox("Normalize", [True, False])
        norm_groups = st.number_input("Norm Groups", min_value=1, max_value=999, value=2)
        num_iters = st.number_input("Number of Iterations", min_value=1, max_value=999, value=5)
        total_reconstruction_loss_weight = st.number_input(
            "Total Reconstruction Loss Weight", min_value=0.0, max_value=1.0, value=0.99999
        )
        total_segmentation_loss_weight = st.number_input(
            "Total Segmentation Loss Weight", min_value=0.0, max_value=1.0, value=0.00001
        )
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": use_reconstruction_module,
            "input_channels": input_channels,
            "reconstruction_module_output_channels": reconstruction_module_output_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "channels": channels,
            "num_pools": num_pools,
            "padding": padding,
            "padding_size": padding_size,
            "drop_prob": drop_prob,
            "normalize": normalize,
            "norm_groups": norm_groups,
            "num_iters": num_iters,
            "total_reconstruction_loss_weight": total_reconstruction_loss_weight,
            "total_segmentation_loss_weight": total_segmentation_loss_weight,
            "use_sens_net": use_sens_net,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Joint MRI Reconstruction and Segmentation (JRS)":
        model_name = "MTLRS"
        joint_reconstruction_segmentation_module_cascades = st.number_input(
            "Joint Reconstruction Segmentation Module Cascades", min_value=1, max_value=999, value=5
        )
        reconstruction_module_recurrent_layer = st.selectbox(
            "Reconstruction Module RNN Type", ["IndRNN", "GRU", "MGU"]
        )
        reconstruction_module_conv_filters = st.text_input(
            "Reconstruction Module Convolutional Filters", value="[128, 128, 2]"
        )
        reconstruction_module_conv_filters = [int(x) for x in reconstruction_module_conv_filters[1:-1].split(",")]
        reconstruction_module_conv_kernels = st.text_input(
            "Reconstruction Module Convolutional Kernels", value="[5, 3, 3]"
        )
        reconstruction_module_conv_kernels = [int(x) for x in reconstruction_module_conv_kernels[1:-1].split(",")]
        reconstruction_module_conv_dilations = st.text_input(
            "Reconstruction Module Convolutional Dilations", value="[1, 2, 1]"
        )
        reconstruction_module_conv_dilations = [int(x) for x in reconstruction_module_conv_dilations[1:-1].split(",")]
        reconstruction_module_conv_bias = st.text_input(
            "Reconstruction Module Convolutional Bias", value=[True, True, False]
        )
        reconstruction_module_recurrent_filters = st.text_input(
            "Reconstruction Module Recurrent Filters", value="[64, 64, 0]"
        )
        reconstruction_module_recurrent_filters = [
            int(x) for x in reconstruction_module_recurrent_filters[1:-1].split(",")
        ]
        reconstruction_module_recurrent_kernels = st.text_input(
            "Reconstruction Module Recurrent Kernels", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_kernels = [
            int(x) for x in reconstruction_module_recurrent_kernels[1:-1].split(",")
        ]
        reconstruction_module_recurrent_dilations = st.text_input(
            "Reconstruction Module Recurrent Dilations", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_dilations = [
            int(x) for x in reconstruction_module_recurrent_dilations[1:-1].split(",")
        ]
        reconstruction_module_recurrent_bias = st.text_input(
            "Reconstruction Module Recurrent Bias", value=[True, True, False]
        )
        reconstruction_module_depth = st.number_input(
            "Reconstruction Module Depth", min_value=1, max_value=999, value=2
        )
        reconstruction_module_time_steps = st.number_input(
            "Reconstruction Module Time Steps", min_value=1, max_value=999, value=8
        )
        reconstruction_module_conv_dim = st.number_input(
            "Reconstruction Module Convolutional Dimension", min_value=1, max_value=3, value=2
        )
        reconstruction_module_num_cascades = st.number_input(
            "Reconstruction Module Number of Cascades", min_value=1, max_value=999, value=1
        )
        reconstruction_module_no_dc = st.selectbox(
            "Reconstruction Module Turn off explicit Data Consistency", [True, False]
        )
        reconstruction_module_keep_prediction = st.selectbox(
            "Reconstruction Module Keep all Predictions over Time Steps", [True, False]
        )
        reconstruction_module_dimensionality = st.number_input(
            "Reconstruction Module Dimensionality", min_value=1, max_value=3, value=2
        )
        reconstruction_module_accumulate_predictions = st.selectbox(
            "Reconstruction Module Accumulate Predictions over Time Steps", [True, False]
        )
        segmentation_module = st.selectbox("Segmentation Module", ["AttentionUNet", "UNet"])
        segmentation_module = segmentation_module.lower()
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        total_reconstruction_loss_weight = st.number_input(
            "Total Reconstruction Loss Weight", min_value=0.0, max_value=1.0, value=0.1
        )
        total_segmentation_loss_weight = st.number_input(
            "Total Segmentation Loss Weight", min_value=0.0, max_value=1.0, value=0.00001
        )
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "joint_reconstruction_segmentation_module_cascades": joint_reconstruction_segmentation_module_cascades,
            "task_adaption_type": None,
            "use_reconstruction_module": True,
            "reconstruction_module_recurrent_layer": reconstruction_module_recurrent_layer,
            "reconstruction_module_conv_filters": reconstruction_module_conv_filters,
            "reconstruction_module_conv_kernels": reconstruction_module_conv_kernels,
            "reconstruction_module_conv_dilations": reconstruction_module_conv_dilations,
            "reconstruction_module_conv_bias": reconstruction_module_conv_bias,
            "reconstruction_module_recurrent_filters": reconstruction_module_recurrent_filters,
            "reconstruction_module_recurrent_kernels": reconstruction_module_recurrent_kernels,
            "reconstruction_module_recurrent_dilations": reconstruction_module_recurrent_dilations,
            "reconstruction_module_recurrent_bias": reconstruction_module_recurrent_bias,
            "reconstruction_module_depth": reconstruction_module_depth,
            "reconstruction_module_time_steps": reconstruction_module_time_steps,
            "reconstruction_module_conv_dim": reconstruction_module_conv_dim,
            "reconstruction_module_num_cascades": reconstruction_module_num_cascades,
            "reconstruction_module_no_dc": reconstruction_module_no_dc,
            "reconstruction_module_keep_prediction": reconstruction_module_keep_prediction,
            "reconstruction_module_dimensionality": reconstruction_module_dimensionality,
            "reconstruction_module_accumulate_predictions": reconstruction_module_accumulate_predictions,
            "segmentation_module": segmentation_module,
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "total_reconstruction_loss_weight": total_reconstruction_loss_weight,
            "total_segmentation_loss_weight": total_segmentation_loss_weight,
            "use_sens_net": use_sens_net,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Multi-Task Learning for MRI Reconstruction and Segmentation (MTLRS)":
        model_name = "MTLRS"
        joint_reconstruction_segmentation_module_cascades = st.number_input(
            "Joint Reconstruction Segmentation Module Cascades", min_value=1, max_value=999, value=5
        )
        reconstruction_module_recurrent_layer = st.selectbox(
            "Reconstruction Module RNN Type", ["IndRNN", "GRU", "MGU"]
        )
        reconstruction_module_conv_filters = st.text_input(
            "Reconstruction Module Convolutional Filters", value="[128, 128, 2]"
        )
        reconstruction_module_conv_filters = [int(x) for x in reconstruction_module_conv_filters[1:-1].split(",")]
        reconstruction_module_conv_kernels = st.text_input(
            "Reconstruction Module Convolutional Kernels", value="[5, 3, 3]"
        )
        reconstruction_module_conv_kernels = [int(x) for x in reconstruction_module_conv_kernels[1:-1].split(",")]
        reconstruction_module_conv_dilations = st.text_input(
            "Reconstruction Module Convolutional Dilations", value="[1, 2, 1]"
        )
        reconstruction_module_conv_dilations = [int(x) for x in reconstruction_module_conv_dilations[1:-1].split(",")]
        reconstruction_module_conv_bias = st.text_input(
            "Reconstruction Module Convolutional Bias", value=[True, True, False]
        )
        reconstruction_module_recurrent_filters = st.text_input(
            "Reconstruction Module Recurrent Filters", value="[64, 64, 0]"
        )
        reconstruction_module_recurrent_filters = [
            int(x) for x in reconstruction_module_recurrent_filters[1:-1].split(",")
        ]
        reconstruction_module_recurrent_kernels = st.text_input(
            "Reconstruction Module Recurrent Kernels", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_kernels = [
            int(x) for x in reconstruction_module_recurrent_kernels[1:-1].split(",")
        ]
        reconstruction_module_recurrent_dilations = st.text_input(
            "Reconstruction Module Recurrent Dilations", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_dilations = [
            int(x) for x in reconstruction_module_recurrent_dilations[1:-1].split(",")
        ]
        reconstruction_module_recurrent_bias = st.text_input(
            "Reconstruction Module Recurrent Bias", value=[True, True, False]
        )
        reconstruction_module_depth = st.number_input(
            "Reconstruction Module Depth", min_value=1, max_value=999, value=2
        )
        reconstruction_module_time_steps = st.number_input(
            "Reconstruction Module Time Steps", min_value=1, max_value=999, value=8
        )
        reconstruction_module_conv_dim = st.number_input(
            "Reconstruction Module Convolutional Dimension", min_value=1, max_value=3, value=2
        )
        reconstruction_module_num_cascades = st.number_input(
            "Reconstruction Module Number of Cascades", min_value=1, max_value=999, value=1
        )
        reconstruction_module_no_dc = st.selectbox(
            "Reconstruction Module Turn off explicit Data Consistency", [True, False]
        )
        reconstruction_module_keep_prediction = st.selectbox(
            "Reconstruction Module Keep all Predictions over Time Steps", [True, False]
        )
        reconstruction_module_dimensionality = st.number_input(
            "Reconstruction Module Dimensionality", min_value=1, max_value=3, value=2
        )
        reconstruction_module_accumulate_predictions = st.selectbox(
            "Reconstruction Module Accumulate Predictions over Time Steps", [True, False]
        )
        segmentation_module = st.selectbox("Segmentation Module", ["AttentionUNet", "UNet"])
        segmentation_module = segmentation_module.lower()
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        total_reconstruction_loss_weight = st.number_input(
            "Total Reconstruction Loss Weight", min_value=0.0, max_value=1.0, value=0.1
        )
        total_segmentation_loss_weight = st.number_input(
            "Total Segmentation Loss Weight", min_value=0.0, max_value=1.0, value=0.00001
        )
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "joint_reconstruction_segmentation_module_cascades": joint_reconstruction_segmentation_module_cascades,
            "task_adaption_type": "multi_task_learning",
            "use_reconstruction_module": True,
            "reconstruction_module_recurrent_layer": reconstruction_module_recurrent_layer,
            "reconstruction_module_conv_filters": reconstruction_module_conv_filters,
            "reconstruction_module_conv_kernels": reconstruction_module_conv_kernels,
            "reconstruction_module_conv_dilations": reconstruction_module_conv_dilations,
            "reconstruction_module_conv_bias": reconstruction_module_conv_bias,
            "reconstruction_module_recurrent_filters": reconstruction_module_recurrent_filters,
            "reconstruction_module_recurrent_kernels": reconstruction_module_recurrent_kernels,
            "reconstruction_module_recurrent_dilations": reconstruction_module_recurrent_dilations,
            "reconstruction_module_recurrent_bias": reconstruction_module_recurrent_bias,
            "reconstruction_module_depth": reconstruction_module_depth,
            "reconstruction_module_time_steps": reconstruction_module_time_steps,
            "reconstruction_module_conv_dim": reconstruction_module_conv_dim,
            "reconstruction_module_num_cascades": reconstruction_module_num_cascades,
            "reconstruction_module_no_dc": reconstruction_module_no_dc,
            "reconstruction_module_keep_prediction": reconstruction_module_keep_prediction,
            "reconstruction_module_dimensionality": reconstruction_module_dimensionality,
            "reconstruction_module_accumulate_predictions": reconstruction_module_accumulate_predictions,
            "segmentation_module": segmentation_module,
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "total_reconstruction_loss_weight": total_reconstruction_loss_weight,
            "total_segmentation_loss_weight": total_segmentation_loss_weight,
            "use_sens_net": use_sens_net,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Reconstruction Segmentation UNet (RecSegUNet)":
        model_name = "RECSEGNET"
        use_reconstruction_module = True
        input_channels = st.number_input("Input Channels", min_value=1, max_value=999, value=1)
        reconstruction_module_output_channels = st.number_input(
            "Reconstruction Module Output Channels", min_value=1, max_value=999, value=1
        )
        reconstruction_module_channels = st.number_input(
            "Reconstruction Module Channels", min_value=1, max_value=999, value=64
        )
        reconstruction_module_pooling_layers = st.number_input(
            "Reconstruction Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        reconstruction_module_dropout = st.number_input(
            "Reconstruction Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        total_reconstruction_loss_weight = st.number_input(
            "Total Reconstruction Loss Weight", min_value=0.0, max_value=1.0, value=0.5
        )
        total_segmentation_loss_weight = st.number_input(
            "Total Segmentation Loss Weight", min_value=0.0, max_value=1.0, value=0.5
        )

        model_args = {
            "use_reconstruction_module": True,
            "input_channels": input_channels,
            "reconstruction_module_output_channels": reconstruction_module_output_channels,
            "reconstruction_module_channels": reconstruction_module_channels,
            "reconstruction_module_pooling_layers": reconstruction_module_pooling_layers,
            "reconstruction_module_dropout": reconstruction_module_dropout,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "total_reconstruction_loss_weight": total_reconstruction_loss_weight,
            "total_segmentation_loss_weight": total_segmentation_loss_weight,
        }
    elif model_name == "Segmentation Network (SegNet)":
        model_name = "SEGNET"
        input_channels = st.number_input("Input Channels (Number of coils x 2)", min_value=1, max_value=999, value=64)
        reconstruction_module_output_channels = st.number_input(
            "Reconstruction Module Output Channels (Number of coils x 2)", min_value=1, max_value=999, value=64
        )
        channels = st.number_input("Reconstruction Module Channels", min_value=1, max_value=999, value=64)
        num_pools = st.number_input("Reconstruction Module Pooling Layers", min_value=1, max_value=999, value=2)
        padding_size = st.number_input("Reconstruction Module Padding Size", min_value=1, max_value=999, value=11)
        drop_prob = st.number_input("Reconstruction Module Dropout", min_value=0.0, max_value=1.0, value=0.0)
        normalize = st.selectbox("Normalize", [False, True])
        num_cascades = st.number_input("Number of Cascades", min_value=1, max_value=999, value=5)
        segmentation_final_layer_conv_dim = st.number_input(
            "Segmentation Final Layer Convolution Dimension", min_value=1, max_value=999, value=2
        )
        segmentation_final_layer_kernel_size = st.number_input(
            "Segmentation Final Layer Kernel Size", min_value=1, max_value=999, value=3
        )
        segmentation_final_layer_dilation = st.number_input(
            "Segmentation Final Layer Dilation", min_value=1, max_value=999, value=1
        )
        segmentation_final_layer_bias = st.selectbox("Segmentation Final Layer Bias", [False, True])
        segmentation_final_layer_nonlinear = st.text_input("Segmentation Final Layer Nonlinear", value="relu")
        total_reconstruction_loss_weight = st.number_input(
            "Total Reconstruction Loss Weight", min_value=0.0, max_value=1.0, value=0.99
        )
        total_segmentation_loss_weight = st.number_input(
            "Total Segmentation Loss Weight", min_value=0.0, max_value=1.0, value=0.01
        )

        model_args = {
            "use_reconstruction_module": True,
            "input_channels": input_channels,
            "reconstruction_module_output_channels": reconstruction_module_output_channels,
            "channels": channels,
            "num_pools": num_pools,
            "padding_size": padding_size,
            "drop_prob": drop_prob,
            "normalize": normalize,
            "num_cascades": num_cascades,
            "segmentation_final_layer_conv_dim": segmentation_final_layer_conv_dim,
            "segmentation_final_layer_kernel_size": segmentation_final_layer_kernel_size,
            "segmentation_final_layer_dilation": segmentation_final_layer_dilation,
            "segmentation_final_layer_bias": segmentation_final_layer_bias,
            "segmentation_final_layer_nonlinear": segmentation_final_layer_nonlinear,
            "total_reconstruction_loss_weight": total_reconstruction_loss_weight,
            "total_segmentation_loss_weight": total_segmentation_loss_weight,
        }
    elif model_name == "Segmentation from k-Space with End-to-End Recurrent Attention Network (SERANet)":
        model_name = "SERANET"
        input_channels = st.number_input("Input Channels", min_value=1, max_value=999, value=2)
        reconstruction_module_output_channels = st.number_input(
            "Reconstruction Module Output Channels", min_value=1, max_value=999, value=2
        )
        reconstruction_module = st.selectbox("Reconstruction Module", ["unet", "cascadenet"])
        if reconstruction_module == "unet":
            reconstruction_module_channels = st.number_input(
                "Reconstruction Module Channels", min_value=1, max_value=999, value=32
            )
            reconstruction_module_pooling_layers = st.number_input(
                "Reconstruction Module Pooling Layers", min_value=1, max_value=999, value=4
            )
            reconstruction_module_dropout = st.number_input(
                "Reconstruction Module Dropout", min_value=0.0, max_value=1.0, value=0.0
            )
        elif reconstruction_module == "cascadenet":
            reconstruction_module_hidden_channels = st.number_input(
                "Reconstruction Module Hidden Channels", min_value=1, max_value=999, value=32
            )
            reconstruction_module_n_convs = st.number_input(
                "Reconstruction Module Number of Convolutions", min_value=1, max_value=999, value=2
            )
            reconstruction_module_batchnorm = st.selectbox("Reconstruction Module Batch Normalization", [True, False])
            reconstruction_module_num_cascades = st.number_input(
                "Reconstruction Module Number of Cascades", min_value=1, max_value=999, value=5
            )
        reconstruction_module_num_blocks = st.number_input(
            "Reconstruction Module Number of Blocks", min_value=1, max_value=999, value=3
        )
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=32
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=32
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=4
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        recurrent_module_iterations = st.number_input(
            "Recurrent Module Iterations", min_value=1, max_value=999, value=2
        )
        recurrent_module_attention_channels = st.number_input(
            "Recurrent Module Attention Channels", min_value=1, max_value=999, value=32
        )
        recurrent_module_attention_pooling_layers = st.number_input(
            "Recurrent Module Attention Pooling Layers", min_value=1, max_value=999, value=4
        )
        recurrent_module_attention_dropout = st.number_input(
            "Recurrent Module Attention Dropout", min_value=0.0, max_value=1.0, value=0.0
        )

        model_args = {
            "use_reconstruction_module": True,
            "input_channels": input_channels,
            "reconstruction_module_output_channels": reconstruction_module_output_channels,
            "reconstruction_module": reconstruction_module,
            "reconstruction_module_num_blocks": reconstruction_module_num_blocks,
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "recurrent_module_iterations": recurrent_module_iterations,
            "recurrent_module_attention_channels": recurrent_module_attention_channels,
            "recurrent_module_attention_pooling_layers": recurrent_module_attention_pooling_layers,
            "recurrent_module_attention_dropout": recurrent_module_attention_dropout,
            "total_reconstruction_loss_weight": 0.0,
            "total_segmentation_loss_weight": 1.0,
        }

        if reconstruction_module == "unet":
            model_args["reconstruction_module_channels"] = reconstruction_module_channels
            model_args["reconstruction_module_pooling_layers"] = reconstruction_module_pooling_layers
            model_args["reconstruction_module_dropout"] = reconstruction_module_dropout
        elif reconstruction_module == "cascadenet":
            model_args["reconstruction_module_hidden_channels"] = reconstruction_module_hidden_channels
            model_args["reconstruction_module_n_convs"] = reconstruction_module_n_convs
            model_args["reconstruction_module_batchnorm"] = reconstruction_module_batchnorm
            model_args["reconstruction_module_num_cascades"] = reconstruction_module_num_cascades
    elif model_name == "quantitative Cascades of Independently Recurrent Inference Machines (qCIRIM)":
        model_name = "qCIRIM"
        use_reconstruction_module = st.selectbox("Use Reconstruction Module", [False, True])
        reconstruction_module_recurrent_layer = st.selectbox(
            "Reconstruction Module RNN Type", ["IndRNN", "GRU", "MGU"]
        )
        reconstruction_module_conv_filters = st.text_input(
            "Reconstruction Module Convolutional Filters", value="[128, 128, 2]"
        )
        reconstruction_module_conv_filters = [int(x) for x in reconstruction_module_conv_filters[1:-1].split(",")]
        reconstruction_module_conv_kernels = st.text_input(
            "Reconstruction Module Convolutional Kernels", value="[5, 3, 3]"
        )
        reconstruction_module_conv_kernels = [int(x) for x in reconstruction_module_conv_kernels[1:-1].split(",")]
        reconstruction_module_conv_dilations = st.text_input(
            "Reconstruction Module Convolutional Dilations", value="[1, 2, 1]"
        )
        reconstruction_module_conv_dilations = [int(x) for x in reconstruction_module_conv_dilations[1:-1].split(",")]
        reconstruction_module_conv_bias = st.text_input(
            "Reconstruction Module Convolutional Bias", value=[True, True, False]
        )
        reconstruction_module_recurrent_filters = st.text_input(
            "Reconstruction Module Recurrent Filters", value="[64, 64, 0]"
        )
        reconstruction_module_recurrent_filters = [
            int(x) for x in reconstruction_module_recurrent_filters[1:-1].split(",")
        ]
        reconstruction_module_recurrent_kernels = st.text_input(
            "Reconstruction Module Recurrent Kernels", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_kernels = [
            int(x) for x in reconstruction_module_recurrent_kernels[1:-1].split(",")
        ]
        reconstruction_module_recurrent_dilations = st.text_input(
            "Reconstruction Module Recurrent Dilations", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_dilations = [
            int(x) for x in reconstruction_module_recurrent_dilations[1:-1].split(",")
        ]
        reconstruction_module_recurrent_bias = st.text_input(
            "Reconstruction Module Recurrent Bias", value=[True, True, False]
        )
        reconstruction_module_depth = st.number_input(
            "Reconstruction Module Depth", min_value=1, max_value=999, value=2
        )
        reconstruction_module_time_steps = st.number_input(
            "Reconstruction Module Time Steps", min_value=1, max_value=999, value=8
        )
        reconstruction_module_conv_dim = st.number_input(
            "Reconstruction Module Convolutional Dimension", min_value=1, max_value=3, value=2
        )
        reconstruction_module_num_cascades = st.number_input(
            "Reconstruction Module Number of Cascades", min_value=1, max_value=999, value=1
        )
        reconstruction_module_no_dc = st.selectbox(
            "Reconstruction Module Turn off explicit Data Consistency", [True, False]
        )
        reconstruction_module_keep_prediction = st.selectbox(
            "Reconstruction Module Keep all Predictions over Time Steps", [True, False]
        )
        reconstruction_module_dimensionality = st.number_input(
            "Reconstruction Module Dimensionality", min_value=1, max_value=3, value=2
        )
        reconstruction_module_accumulate_predictions = st.selectbox(
            "Reconstruction Module Accumulate Predictions over Time Steps", [True, False]
        )
        quantitative_module_recurrent_layer = st.selectbox("Quantitative Module RNN Type", ["IndRNN", "GRU", "MGU"])
        quantitative_module_conv_filters = st.text_input(
            "Quantitative Module Convolutional Filters", value="[128, 128, 2]"
        )
        quantitative_module_conv_filters = [int(x) for x in quantitative_module_conv_filters[1:-1].split(",")]
        quantitative_module_conv_kernels = st.text_input(
            "Quantitative Module Convolutional Kernels", value="[5, 3, 3]"
        )
        quantitative_module_conv_kernels = [int(x) for x in quantitative_module_conv_kernels[1:-1].split(",")]
        quantitative_module_conv_dilations = st.text_input(
            "Quantitative Module Convolutional Dilations", value="[1, 2, 1]"
        )
        quantitative_module_conv_dilations = [int(x) for x in quantitative_module_conv_dilations[1:-1].split(",")]
        quantitative_module_conv_bias = st.text_input(
            "Quantitative Module Convolutional Bias", value=[True, True, False]
        )
        quantitative_module_recurrent_filters = st.text_input(
            "Quantitative Module Recurrent Filters", value="[64, 64, 0]"
        )
        quantitative_module_recurrent_filters = [
            int(x) for x in quantitative_module_recurrent_filters[1:-1].split(",")
        ]
        quantitative_module_recurrent_kernels = st.text_input(
            "Quantitative Module Recurrent Kernels", value="[1, 1, 0]"
        )
        quantitative_module_recurrent_kernels = [
            int(x) for x in quantitative_module_recurrent_kernels[1:-1].split(",")
        ]
        quantitative_module_recurrent_dilations = st.text_input(
            "Quantitative Module Recurrent Dilations", value="[1, 1, 0]"
        )
        quantitative_module_recurrent_dilations = [
            int(x) for x in quantitative_module_recurrent_dilations[1:-1].split(",")
        ]
        quantitative_module_recurrent_bias = st.text_input(
            "Quantitative Module Recurrent Bias", value=[True, True, False]
        )
        quantitative_module_depth = st.number_input("Quantitative Module Depth", min_value=1, max_value=999, value=2)
        quantitative_module_time_steps = st.number_input(
            "Quantitative Module Time Steps", min_value=1, max_value=999, value=8
        )
        quantitative_module_conv_dim = st.number_input(
            "Quantitative Module Convolutional Dimension", min_value=1, max_value=3, value=2
        )
        quantitative_module_num_cascades = st.number_input(
            "Quantitative Module Number of Cascades", min_value=1, max_value=999, value=5
        )
        quantitative_module_no_dc = st.selectbox(
            "Quantitative Module Turn off explicit Data Consistency", [True, False]
        )
        quantitative_module_keep_prediction = st.selectbox(
            "Quantitative Module Keep all Predictions over Time Steps", [True, False]
        )
        quantitative_module_dimensionality = st.number_input(
            "Quantitative Module Dimensionality", min_value=1, max_value=3, value=2
        )
        quantitative_module_accumulate_predictions = st.selectbox(
            "Quantitative Module Accumulate Predictions over Time Steps", [True, False]
        )
        quantitative_module_signal_forward_model_sequence = st.selectbox(
            "Quantitative Module Signal Forward Model Sequence", ["MEGRE"]
        )
        quantitative_module_gamma_regularization_factors = st.text_input(
            "Quantitative Module Gamma Regularization Factors", value="[150.0, 150.0, 1000.0, 150.0]"
        )
        shift_B0_input = st.selectbox("Shift B0 Input", [False, True])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "use_reconstruction_module": use_reconstruction_module,
            "reconstruction_module_recurrent_layer": reconstruction_module_recurrent_layer,
            "reconstruction_module_conv_filters": reconstruction_module_conv_filters,
            "reconstruction_module_conv_kernels": reconstruction_module_conv_kernels,
            "reconstruction_module_conv_dilations": reconstruction_module_conv_dilations,
            "reconstruction_module_conv_bias": reconstruction_module_conv_bias,
            "reconstruction_module_recurrent_filters": reconstruction_module_recurrent_filters,
            "reconstruction_module_recurrent_kernels": reconstruction_module_recurrent_kernels,
            "reconstruction_module_recurrent_dilations": reconstruction_module_recurrent_dilations,
            "reconstruction_module_recurrent_bias": reconstruction_module_recurrent_bias,
            "reconstruction_module_depth": reconstruction_module_depth,
            "reconstruction_module_time_steps": reconstruction_module_time_steps,
            "reconstruction_module_conv_dim": reconstruction_module_conv_dim,
            "reconstruction_module_num_cascades": reconstruction_module_num_cascades,
            "reconstruction_module_no_dc": reconstruction_module_no_dc,
            "reconstruction_module_keep_prediction": reconstruction_module_keep_prediction,
            "reconstruction_module_dimensionality": reconstruction_module_dimensionality,
            "reconstruction_module_accumulate_predictions": reconstruction_module_accumulate_predictions,
            "quantitative_module_recurrent_layer": quantitative_module_recurrent_layer,
            "quantitative_module_conv_filters": quantitative_module_conv_filters,
            "quantitative_module_conv_kernels": quantitative_module_conv_kernels,
            "quantitative_module_conv_dilations": quantitative_module_conv_dilations,
            "quantitative_module_conv_bias": quantitative_module_conv_bias,
            "quantitative_module_recurrent_filters": quantitative_module_recurrent_filters,
            "quantitative_module_recurrent_kernels": quantitative_module_recurrent_kernels,
            "quantitative_module_recurrent_dilations": quantitative_module_recurrent_dilations,
            "quantitative_module_recurrent_bias": quantitative_module_recurrent_bias,
            "quantitative_module_depth": quantitative_module_depth,
            "quantitative_module_time_steps": quantitative_module_time_steps,
            "quantitative_module_conv_dim": quantitative_module_conv_dim,
            "quantitative_module_num_cascades": quantitative_module_num_cascades,
            "quantitative_module_no_dc": quantitative_module_no_dc,
            "quantitative_module_keep_prediction": quantitative_module_keep_prediction,
            "quantitative_module_dimensionality": quantitative_module_dimensionality,
            "quantitative_module_accumulate_predictions": quantitative_module_accumulate_predictions,
            "quantitative_module_signal_forward_model_sequence": quantitative_module_signal_forward_model_sequence,
            "quantitative_module_gamma_regularization_factors": quantitative_module_gamma_regularization_factors,
            "shift_B0_input": shift_B0_input,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "quantitative Recurrent Inference Machines (qRIM)":
        model_name = "qCIRIM"
        rim_only = True
        use_reconstruction_module = st.selectbox("Use Reconstruction Module", [False, True])
        reconstruction_module_recurrent_layer = st.selectbox(
            "Reconstruction Module RNN Type", ["GRU", "IndRNN", "MGU"]
        )
        reconstruction_module_conv_filters = st.text_input(
            "Reconstruction Module Convolutional Filters", value="[128, 128, 2]"
        )
        reconstruction_module_conv_filters = [int(x) for x in reconstruction_module_conv_filters[1:-1].split(",")]
        reconstruction_module_conv_kernels = st.text_input(
            "Reconstruction Module Convolutional Kernels", value="[5, 3, 3]"
        )
        reconstruction_module_conv_kernels = [int(x) for x in reconstruction_module_conv_kernels[1:-1].split(",")]
        reconstruction_module_conv_dilations = st.text_input(
            "Reconstruction Module Convolutional Dilations", value="[1, 2, 1]"
        )
        reconstruction_module_conv_dilations = [int(x) for x in reconstruction_module_conv_dilations[1:-1].split(",")]
        reconstruction_module_conv_bias = st.text_input(
            "Reconstruction Module Convolutional Bias", value=[True, True, False]
        )
        reconstruction_module_recurrent_filters = st.text_input(
            "Reconstruction Module Recurrent Filters", value="[64, 64, 0]"
        )
        reconstruction_module_recurrent_filters = [
            int(x) for x in reconstruction_module_recurrent_filters[1:-1].split(",")
        ]
        reconstruction_module_recurrent_kernels = st.text_input(
            "Reconstruction Module Recurrent Kernels", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_kernels = [
            int(x) for x in reconstruction_module_recurrent_kernels[1:-1].split(",")
        ]
        reconstruction_module_recurrent_dilations = st.text_input(
            "Reconstruction Module Recurrent Dilations", value="[1, 1, 0]"
        )
        reconstruction_module_recurrent_dilations = [
            int(x) for x in reconstruction_module_recurrent_dilations[1:-1].split(",")
        ]
        reconstruction_module_recurrent_bias = st.text_input(
            "Reconstruction Module Recurrent Bias", value=[True, True, False]
        )
        reconstruction_module_depth = st.number_input(
            "Reconstruction Module Depth", min_value=1, max_value=999, value=2
        )
        reconstruction_module_time_steps = st.number_input(
            "Reconstruction Module Time Steps", min_value=1, max_value=999, value=8
        )
        reconstruction_module_conv_dim = st.number_input(
            "Reconstruction Module Convolutional Dimension", min_value=1, max_value=3, value=2
        )
        reconstruction_module_num_cascades = 1
        reconstruction_module_no_dc = st.selectbox(
            "Reconstruction Module Turn off explicit Data Consistency", [True, False]
        )
        reconstruction_module_keep_prediction = st.selectbox(
            "Reconstruction Module Keep all Predictions over Time Steps", [True, False]
        )
        reconstruction_module_dimensionality = st.number_input(
            "Reconstruction Module Dimensionality", min_value=1, max_value=3, value=2
        )
        reconstruction_module_accumulate_predictions = st.selectbox(
            "Reconstruction Module Accumulate Predictions over Time Steps", [True, False]
        )
        quantitative_module_recurrent_layer = st.selectbox("Quantitative Module RNN Type", ["GRU", "IndRNN", "MGU"])
        quantitative_module_conv_filters = st.text_input(
            "Quantitative Module Convolutional Filters", value="[128, 128, 2]"
        )
        quantitative_module_conv_filters = [int(x) for x in quantitative_module_conv_filters[1:-1].split(",")]
        quantitative_module_conv_kernels = st.text_input(
            "Quantitative Module Convolutional Kernels", value="[5, 3, 3]"
        )
        quantitative_module_conv_kernels = [int(x) for x in quantitative_module_conv_kernels[1:-1].split(",")]
        quantitative_module_conv_dilations = st.text_input(
            "Quantitative Module Convolutional Dilations", value="[1, 2, 1]"
        )
        quantitative_module_conv_dilations = [int(x) for x in quantitative_module_conv_dilations[1:-1].split(",")]
        quantitative_module_conv_bias = st.text_input(
            "Quantitative Module Convolutional Bias", value=[True, True, False]
        )
        quantitative_module_recurrent_filters = st.text_input(
            "Quantitative Module Recurrent Filters", value="[64, 64, 0]"
        )
        quantitative_module_recurrent_filters = [
            int(x) for x in quantitative_module_recurrent_filters[1:-1].split(",")
        ]
        quantitative_module_recurrent_kernels = st.text_input(
            "Quantitative Module Recurrent Kernels", value="[1, 1, 0]"
        )
        quantitative_module_recurrent_kernels = [
            int(x) for x in quantitative_module_recurrent_kernels[1:-1].split(",")
        ]
        quantitative_module_recurrent_dilations = st.text_input(
            "Quantitative Module Recurrent Dilations", value="[1, 1, 0]"
        )
        quantitative_module_recurrent_dilations = [
            int(x) for x in quantitative_module_recurrent_dilations[1:-1].split(",")
        ]
        quantitative_module_recurrent_bias = st.text_input(
            "Quantitative Module Recurrent Bias", value=[True, True, False]
        )
        quantitative_module_depth = st.number_input("Quantitative Module Depth", min_value=1, max_value=999, value=2)
        quantitative_module_time_steps = st.number_input(
            "Quantitative Module Time Steps", min_value=1, max_value=999, value=8
        )
        quantitative_module_conv_dim = st.number_input(
            "Quantitative Module Convolutional Dimension", min_value=1, max_value=3, value=2
        )
        quantitative_module_num_cascades = 1
        quantitative_module_no_dc = st.selectbox(
            "Quantitative Module Turn off explicit Data Consistency", [True, False]
        )
        quantitative_module_keep_prediction = st.selectbox(
            "Quantitative Module Keep all Predictions over Time Steps", [True, False]
        )
        quantitative_module_dimensionality = st.number_input(
            "Quantitative Module Dimensionality", min_value=1, max_value=3, value=2
        )
        quantitative_module_accumulate_predictions = st.selectbox(
            "Quantitative Module Accumulate Predictions over Time Steps", [True, False]
        )
        quantitative_module_signal_forward_model_sequence = st.selectbox(
            "Quantitative Module Signal Forward Model Sequence", ["MEGRE"]
        )
        quantitative_module_gamma_regularization_factors = st.text_input(
            "Quantitative Module Gamma Regularization Factors", value="[150.0, 150.0, 1000.0, 150.0]"
        )
        shift_B0_input = st.selectbox("Shift B0 Input", [False, True])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "use_reconstruction_module": use_reconstruction_module,
            "reconstruction_module_recurrent_layer": reconstruction_module_recurrent_layer,
            "reconstruction_module_conv_filters": reconstruction_module_conv_filters,
            "reconstruction_module_conv_kernels": reconstruction_module_conv_kernels,
            "reconstruction_module_conv_dilations": reconstruction_module_conv_dilations,
            "reconstruction_module_conv_bias": reconstruction_module_conv_bias,
            "reconstruction_module_recurrent_filters": reconstruction_module_recurrent_filters,
            "reconstruction_module_recurrent_kernels": reconstruction_module_recurrent_kernels,
            "reconstruction_module_recurrent_dilations": reconstruction_module_recurrent_dilations,
            "reconstruction_module_recurrent_bias": reconstruction_module_recurrent_bias,
            "reconstruction_module_depth": reconstruction_module_depth,
            "reconstruction_module_time_steps": reconstruction_module_time_steps,
            "reconstruction_module_conv_dim": reconstruction_module_conv_dim,
            "reconstruction_module_num_cascades": reconstruction_module_num_cascades,
            "reconstruction_module_no_dc": reconstruction_module_no_dc,
            "reconstruction_module_keep_prediction": reconstruction_module_keep_prediction,
            "reconstruction_module_dimensionality": reconstruction_module_dimensionality,
            "reconstruction_module_accumulate_predictions": reconstruction_module_accumulate_predictions,
            "quantitative_module_recurrent_layer": quantitative_module_recurrent_layer,
            "quantitative_module_conv_filters": quantitative_module_conv_filters,
            "quantitative_module_conv_kernels": quantitative_module_conv_kernels,
            "quantitative_module_conv_dilations": quantitative_module_conv_dilations,
            "quantitative_module_conv_bias": quantitative_module_conv_bias,
            "quantitative_module_recurrent_filters": quantitative_module_recurrent_filters,
            "quantitative_module_recurrent_kernels": quantitative_module_recurrent_kernels,
            "quantitative_module_recurrent_dilations": quantitative_module_recurrent_dilations,
            "quantitative_module_recurrent_bias": quantitative_module_recurrent_bias,
            "quantitative_module_depth": quantitative_module_depth,
            "quantitative_module_time_steps": quantitative_module_time_steps,
            "quantitative_module_conv_dim": quantitative_module_conv_dim,
            "quantitative_module_num_cascades": quantitative_module_num_cascades,
            "quantitative_module_no_dc": quantitative_module_no_dc,
            "quantitative_module_keep_prediction": quantitative_module_keep_prediction,
            "quantitative_module_dimensionality": quantitative_module_dimensionality,
            "quantitative_module_accumulate_predictions": quantitative_module_accumulate_predictions,
            "quantitative_module_signal_forward_model_sequence": quantitative_module_signal_forward_model_sequence,
            "quantitative_module_gamma_regularization_factors": quantitative_module_gamma_regularization_factors,
            "shift_B0_input": shift_B0_input,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "quantitative Variational Network (qVN)":
        model_name = "qVN"
        use_reconstruction_module = st.selectbox("Use Reconstruction Module", [False, True])
        reconstruction_module_num_cascades = st.number_input(
            "Reconstruction Module Number of Cascades", min_value=1, max_value=999, value=2
        )
        reconstruction_module_channels = st.number_input(
            "Reconstruction Module Channels", min_value=1, max_value=999, value=8
        )
        reconstruction_module_pooling_layers = st.number_input(
            "Reconstruction Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        reconstruction_module_in_channels = st.number_input(
            "Reconstruction Module Input Channels", min_value=1, max_value=999, value=2
        )
        reconstruction_module_out_channels = st.number_input(
            "Reconstruction Module Output Channels", min_value=1, max_value=999, value=2
        )
        reconstruction_module_padding_size = st.number_input(
            "Reconstruction Module Padding Size", min_value=1, max_value=999, value=11
        )
        reconstruction_module_normalize = st.selectbox("Reconstruction Module Normalize", [True, False])
        reconstruction_module_no_dc = st.selectbox(
            "Reconstruction Module Turn off Explicit Data Consistency", [False, True]
        )
        reconstruction_module_dimensionality = st.number_input(
            "Reconstruction Module Dimensionality", min_value=1, max_value=999, value=2
        )
        reconstruction_module_accumulate_predictions = st.selectbox(
            "Reconstruction Module Accumulate Predictions", [False, True]
        )
        quantitative_module_num_cascades = st.number_input(
            "Quantitative Module Number of Cascades", min_value=1, max_value=999, value=2
        )
        quantitative_module_channels = st.number_input(
            "Quantitative Module Channels", min_value=1, max_value=999, value=8
        )
        quantitative_module_pooling_layers = st.number_input(
            "Quantitative Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        quantitative_module_in_channels = st.number_input(
            "Quantitative Module Input Channels", min_value=1, max_value=999, value=2
        )
        quantitative_module_out_channels = st.number_input(
            "Quantitative Module Output Channels", min_value=1, max_value=999, value=2
        )
        quantitative_module_padding_size = st.number_input(
            "Quantitative Module Padding Size", min_value=1, max_value=999, value=11
        )
        quantitative_module_normalize = st.selectbox("Quantitative Module Normalize", [True, False])
        quantitative_module_no_dc = st.selectbox(
            "Quantitative Module Turn off explicit Data Consistency", [False, True]
        )
        quantitative_module_dimensionality = st.number_input(
            "Quantitative Module Dimensionality", min_value=1, max_value=3, value=2
        )
        quantitative_module_signal_forward_model_sequence = st.selectbox(
            "Quantitative Module Signal Forward Model Sequence", ["MEGRE"]
        )
        quantitative_module_accumulate_predictions = st.selectbox(
            "Quantitative Module Accumulate Predictions over Time Steps", [True, False]
        )
        quantitative_module_gamma_regularization_factors = st.text_input(
            "Quantitative Module Gamma Regularization Factors", value="[150.0, 150.0, 1000.0, 150.0]"
        )
        shift_B0_input = st.selectbox("Shift B0 Input", [False, True])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "use_reconstruction_module": use_reconstruction_module,
            "reconstruction_module_num_cascades": reconstruction_module_num_cascades,
            "reconstruction_module_channels": reconstruction_module_channels,
            "reconstruction_module_pooling_layers": reconstruction_module_pooling_layers,
            "reconstruction_module_in_channels": reconstruction_module_in_channels,
            "reconstruction_module_out_channels": reconstruction_module_out_channels,
            "reconstruction_module_padding_size": reconstruction_module_padding_size,
            "reconstruction_module_normalize": reconstruction_module_normalize,
            "reconstruction_module_no_dc": reconstruction_module_no_dc,
            "reconstruction_module_dimensionality": reconstruction_module_dimensionality,
            "reconstruction_module_accumulate_predictions": reconstruction_module_accumulate_predictions,
            "quantitative_module_num_cascades": quantitative_module_num_cascades,
            "quantitative_module_channels": quantitative_module_channels,
            "quantitative_module_pooling_layers": quantitative_module_pooling_layers,
            "quantitative_module_in_channels": quantitative_module_in_channels,
            "quantitative_module_out_channels": quantitative_module_out_channels,
            "quantitative_module_padding_size": quantitative_module_padding_size,
            "quantitative_module_normalize": quantitative_module_normalize,
            "quantitative_module_no_dc": quantitative_module_no_dc,
            "quantitative_module_dimensionality": quantitative_module_dimensionality,
            "quantitative_module_signal_forward_model_sequence": quantitative_module_signal_forward_model_sequence,
            "quantitative_module_accumulate_predictions": quantitative_module_accumulate_predictions,
            "quantitative_module_gamma_regularization_factors": quantitative_module_gamma_regularization_factors,
            "shift_B0_input": shift_B0_input,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Cascades of Independently Recurrent Inference Machines (CIRIM)":
        model_name = "CIRIM"
        recurrent_layer = st.selectbox("RNN Type", ["IndRNN", "GRU", "MGU"])
        conv_filters = st.text_input("Convolutional Filters", value="[64, 64, 2]")
        conv_filters = [int(x) for x in conv_filters[1:-1].split(",")]
        conv_kernels = st.text_input("Convolutional Kernels", value="[5, 3, 3]")
        conv_kernels = [int(x) for x in conv_kernels[1:-1].split(",")]
        conv_dilations = st.text_input("Convolutional Dilations", value="[1, 2, 1]")
        conv_dilations = [int(x) for x in conv_dilations[1:-1].split(",")]
        conv_bias = st.text_input("Convolutional Bias", value=[True, True, False])
        recurrent_filters = st.text_input("Recurrent Filters", value="[64, 64, 0]")
        recurrent_filters = [int(x) for x in recurrent_filters[1:-1].split(",")]
        recurrent_kernels = st.text_input("Recurrent Kernels", value="[1, 1, 0]")
        recurrent_kernels = [int(x) for x in recurrent_kernels[1:-1].split(",")]
        recurrent_dilations = st.text_input("Recurrent Dilations", value="[1, 1, 0]")
        recurrent_dilations = [int(x) for x in recurrent_dilations[1:-1].split(",")]
        recurrent_bias = st.text_input("Recurrent Bias", value=[True, True, False])
        depth = st.number_input("Depth", min_value=1, max_value=999, value=2)
        time_steps = st.number_input("Time Steps", min_value=1, max_value=999, value=8)
        conv_dim = st.number_input("Convolutional Dimension", min_value=1, max_value=3, value=2)
        num_cascades = st.number_input("Number of Cascades", min_value=1, max_value=999, value=5)
        no_dc = st.selectbox("Turn off explicit Data Consistency", [True, False])
        keep_prediction = st.selectbox("Keep all Predictions over Time Steps", [True, False])
        accumulate_predictions = st.selectbox("Accumulate Predictions over Time Steps", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "recurrent_layer": recurrent_layer,
            "conv_filters": conv_filters,
            "conv_kernels": conv_kernels,
            "conv_dilations": conv_dilations,
            "conv_bias": conv_bias,
            "recurrent_filters": recurrent_filters,
            "recurrent_kernels": recurrent_kernels,
            "recurrent_dilations": recurrent_dilations,
            "recurrent_bias": recurrent_bias,
            "depth": depth,
            "time_steps": time_steps,
            "conv_dim": conv_dim,
            "num_cascades": num_cascades,
            "no_dc": no_dc,
            "keep_prediction": keep_prediction,
            "accumulate_predictions": accumulate_predictions,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Convolutional Recurrent Neural Network (CRNN)":
        model_name = "CRNNET"
        num_iterations = st.number_input("Number of Iterations", min_value=1, max_value=999, value=10)
        hidden_channels = st.number_input("Hidden Channels", min_value=1, max_value=999, value=64)
        n_convs = st.number_input("Number of Convolutions", min_value=1, max_value=999, value=3)
        batchnorm = st.selectbox("Batch Normalization", [False, True])
        no_dc = st.selectbox("Turn off explicit Data Consistency", [False, True])
        accumulate_predictions = st.selectbox("Accumulate Predictions over Time Steps", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "num_iterations": num_iterations,
            "hidden_channels": hidden_channels,
            "n_convs": n_convs,
            "batchnorm": batchnorm,
            "no_dc": no_dc,
            "accumulate_predictions": accumulate_predictions,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Deep Cascade of Convolutional Neural Networks (DCCNN)":
        model_name = "CASCADENET"
        num_cascades = st.number_input("Number of Cascades", min_value=1, max_value=999, value=10)
        hidden_channels = st.number_input("Hidden Channels", min_value=1, max_value=999, value=64)
        n_convs = st.number_input("Number of Convolutions", min_value=1, max_value=999, value=5)
        batchnorm = st.selectbox("Batch Normalization", [False, True])
        no_dc = st.selectbox("Turn off explicit Data Consistency", [False, True])
        accumulate_predictions = st.selectbox("Accumulate Predictions over Time Steps", [False, True])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "num_cascades": num_cascades,
            "hidden_channels": hidden_channels,
            "n_convs": n_convs,
            "batchnorm": batchnorm,
            "no_dc": no_dc,
            "accumulate_predictions": accumulate_predictions,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Down-Up NET (DUNET)":
        model_name = "DUNET"
        num_iter = st.number_input("Number of Iterations", min_value=1, max_value=999, value=10)
        reg_model_architecture = st.selectbox("Regularization Model Architecture", ["DIDN", "UNet"])
        in_channels = st.number_input("Input Channels", min_value=1, max_value=999, value=1)
        out_channels = st.number_input("Output Channels", min_value=1, max_value=999, value=1)
        if reg_model_architecture == "DIDN":
            hidden_channels = st.number_input("Hidden Channels", min_value=1, max_value=999, value=64)
            num_dubs = st.number_input("Number of Down-Up Blocks", min_value=1, max_value=999, value=2)
            num_convs_recon = st.number_input(
                "Number of Convolutions in Reconstruction", min_value=1, max_value=999, value=1
            )
        else:
            unet_num_filters = st.number_input("Number of Filters", min_value=1, max_value=999, value=64)
            unet_num_pool_layers = st.number_input("Number of Pooling Layers", min_value=1, max_value=999, value=4)
            drop_prob = st.number_input("Dropout Probability", min_value=0.0, max_value=1.0, value=0.0)
            padding_size = st.number_input("Padding Size", min_value=0, max_value=999, value=11)
            normalize = st.selectbox("Normalize", [True, False])
        data_consistency_term = st.selectbox("Data Consistency Term", ["GD", "PROX", "VS", "ID"])
        if data_consistency_term in ["GD", "PROX"]:
            data_consistency_lambda_init = st.number_input(
                "Data Consistency Regularization factor", min_value=0.0, max_value=999.0, value=0.1
            )
        if data_consistency_term == "PROX":
            data_consistency_iterations = st.number_input(
                "Data Consistency Iterations", min_value=1, max_value=999, value=10
            )
        if data_consistency_term == "VS":
            alpha_init = st.number_input("Alpha factor", min_value=0.0, max_value=999.0, value=0.1)
            beta_init = st.number_input("Beta factor", min_value=0.0, max_value=999.0, value=0.1)
        shared_params = st.selectbox("Shared Parameters", [False, True])
        use_sens_net = False

        model_args = {
            "num_iter": num_iter,
            "reg_model_architecture": reg_model_architecture.upper(),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "shared_params": shared_params,
            "use_sens_net": use_sens_net,
        }

        if reg_model_architecture == "DIDN":
            model_args["hidden_channels"] = hidden_channels
            model_args["num_dubs"] = num_dubs
            model_args["num_convs_recon"] = num_convs_recon
        else:
            model_args["unet_num_filters"] = unet_num_filters
            model_args["unet_num_pool_layers"] = unet_num_pool_layers
            model_args["drop_prob"] = drop_prob
            model_args["padding_size"] = padding_size
            model_args["normalize"] = normalize

        if data_consistency_term in ["GD"]:
            model_args["data_consistency_lambda_init"] = data_consistency_lambda_init
        elif data_consistency_term == "PROX":
            model_args["data_consistency_lambda_init"] = data_consistency_lambda_init
            model_args["data_consistency_iterations"] = data_consistency_iterations
        elif data_consistency_term == "VS":
            model_args["alpha_init"] = alpha_init
            model_args["beta_init"] = beta_init
    elif model_name == "Feature-level multi-domain network (MultiDomainNet)":
        model_name = "MULTIDOMAINNET"
        standardization = st.selectbox("Standardization", [True, False])
        num_filters = st.number_input("Number of Filters", min_value=1, max_value=999, value=64)
        num_pool_layers = st.number_input("Number of Pooling Layers", min_value=1, max_value=999, value=4)
        dropout_probability = st.number_input("Dropout Probability", min_value=0.0, max_value=1.0, value=0.0)
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "num_cascades": standardization,
            "num_filters": num_filters,
            "num_pool_layers": num_pool_layers,
            "dropout_probability": dropout_probability,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet)":
        model_name = "JOINTICNET"
        num_iter = st.number_input("Number of Iterations", min_value=1, max_value=999, value=2)
        kspace_unet_num_filters = st.number_input("k-space UNet filters", min_value=1, max_value=999, value=16)
        kspace_unet_num_pool_layers = st.number_input(
            "k-space UNet pooling layers", min_value=1, max_value=999, value=2
        )
        kspace_unet_dropout_probability = st.number_input(
            "k-space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
        )
        kspace_unet_padding_size = st.number_input("k-space UNet padding size", min_value=0, max_value=999, value=11)
        kspace_unet_normalize = st.selectbox("k-space UNet normalize", [True, False])
        imspace_unet_num_filters = st.number_input("Image space UNet filters", min_value=1, max_value=999, value=16)
        imspace_unet_num_pool_layers = st.number_input(
            "Image space UNet pooling layers", min_value=1, max_value=999, value=2
        )
        imspace_unet_dropout_probability = st.number_input(
            "Image space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
        )
        imspace_unet_padding_size = st.number_input(
            "Image space UNet padding size", min_value=0, max_value=999, value=11
        )
        imspace_unet_normalize = st.selectbox("Image space UNet normalize", [True, False])
        sens_unet_num_filters = st.number_input("Sensitivity UNet filters", min_value=1, max_value=999, value=16)
        sens_unet_num_pool_layers = st.number_input(
            "Sensitivity UNet pooling layers", min_value=1, max_value=999, value=2
        )
        sens_unet_dropout_probability = st.number_input(
            "Sensitivity UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
        )
        sens_unet_padding_size = st.number_input("Sensitivity UNet padding size", min_value=0, max_value=999, value=11)
        sens_unet_normalize = st.selectbox("Sensitivity UNet normalize", [True, False])

        model_args = {
            "num_iter": num_iter,
            "kspace_unet_num_filters": kspace_unet_num_filters,
            "kspace_unet_num_pool_layers": kspace_unet_num_pool_layers,
            "kspace_unet_dropout_probability": kspace_unet_dropout_probability,
            "kspace_unet_padding_size": kspace_unet_padding_size,
            "kspace_unet_normalize": kspace_unet_normalize,
            "imspace_unet_num_filters": imspace_unet_num_filters,
            "imspace_unet_num_pool_layers": imspace_unet_num_pool_layers,
            "imspace_unet_dropout_probability": imspace_unet_dropout_probability,
            "imspace_unet_padding_size": imspace_unet_padding_size,
            "imspace_unet_normalize": imspace_unet_normalize,
            "sens_unet_num_filters": sens_unet_num_filters,
            "sens_unet_num_pool_layers": sens_unet_num_pool_layers,
            "sens_unet_dropout_probability": sens_unet_dropout_probability,
            "sens_unet_padding_size": sens_unet_padding_size,
            "sens_unet_normalize": sens_unet_normalize,
        }
    elif model_name == "KIKI-Net: Cross-Domain Convolutional Neural Networks":
        model_name = "KIKINET"
        num_iter = st.number_input("Number of Iterations", min_value=1, max_value=999, value=2)
        kspace_model_architecture = st.selectbox("k-space Model Architecture", ["UNET", "CONV", "DIDN"])
        kspace_in_channels = st.number_input("k-space Input Channels", min_value=1, max_value=999, value=2)
        kspace_out_channels = st.number_input("k-space Output Channels", min_value=1, max_value=999, value=2)
        if kspace_model_architecture == "UNET":
            kspace_unet_num_filters = st.number_input("k-space UNet filters", min_value=1, max_value=999, value=16)
            kspace_unet_num_pool_layers = st.number_input(
                "k-space UNet pooling layers", min_value=1, max_value=999, value=2
            )
            kspace_unet_dropout_probability = st.number_input(
                "k-space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            kspace_unet_padding_size = st.number_input(
                "k-space UNet padding size", min_value=0, max_value=999, value=11
            )
            kspace_unet_normalize = st.selectbox("k-space UNet normalize", [True, False])
        elif kspace_model_architecture == "DIDN":
            kspace_didn_hidden_channels = st.number_input(
                "k-space DIDN hidden channels", min_value=1, max_value=999, value=64
            )
            num_dubs = st.number_input("Number of Doubles", min_value=1, max_value=999, value=2)
            num_convs_recon = st.number_input(
                "Number of Convolutions in Reconstruction", min_value=1, max_value=999, value=2
            )
        elif kspace_model_architecture == "CONV":
            kspace_conv_hidden_channels = st.number_input(
                "k-space Convolutional hidden channels", min_value=1, max_value=999, value=64
            )
            kspace_conv_n_convs = st.number_input(
                "k-space Convolutional number of convolutions", min_value=1, max_value=999, value=2
            )
            kspace_conv_batchnorm = st.selectbox("k-space Convolutional batchnorm", [True, False])
        imspace_model_architecture = st.selectbox("Image space Model Architecture", ["UNET", "MWCNN"])
        imspace_in_channels = st.number_input("Image space Input Channels", min_value=1, max_value=999, value=2)
        imspace_out_channels = st.number_input("Image space Output Channels", min_value=1, max_value=999, value=2)
        if imspace_model_architecture == "UNET":
            imspace_unet_num_filters = st.number_input(
                "Image space UNet filters", min_value=1, max_value=999, value=16
            )
            imspace_unet_num_pool_layers = st.number_input(
                "Image space UNet pooling layers", min_value=1, max_value=999, value=2
            )
            imspace_unet_dropout_probability = st.number_input(
                "Image space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            imspace_unet_padding_size = st.number_input(
                "Image space UNet padding size", min_value=0, max_value=999, value=11
            )
            imspace_unet_normalize = st.selectbox("Image space UNet normalize", [True, False])
        elif imspace_model_architecture == "MWCNN":
            image_mwcnn_hidden_channels = st.number_input(
                "Image space MWCNN hidden channels", min_value=1, max_value=999, value=64
            )
            num_scales = st.number_input("Number of Scales", min_value=1, max_value=999, value=2)
            bias = st.selectbox("Bias", [True, False])
            batchnorm = st.selectbox("Batchnorm", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Net", [False, True])

        model_args = {
            "num_iter": num_iter,
            "kspace_model_architecture": kspace_model_architecture,
            "kspace_in_channels": kspace_in_channels,
            "kspace_out_channels": kspace_out_channels,
            "imspace_model_architecture": imspace_model_architecture,
            "imspace_in_channels": imspace_in_channels,
            "imspace_out_channels": imspace_out_channels,
            "use_sens_net": use_sens_net,
        }
        if kspace_model_architecture == "UNET":
            model_args["kspace_unet_num_filters"] = kspace_unet_num_filters
            model_args["kspace_unet_num_pool_layers"] = kspace_unet_num_pool_layers
            model_args["kspace_unet_dropout_probability"] = kspace_unet_dropout_probability
            model_args["kspace_unet_padding_size"] = kspace_unet_padding_size
            model_args["kspace_unet_normalize"] = kspace_unet_normalize
        elif kspace_model_architecture == "DIDN":
            model_args["kspace_didn_hidden_channels"] = kspace_didn_hidden_channels
            model_args["num_dubs"] = num_dubs
            model_args["num_convs_recon"] = num_convs_recon
        elif kspace_model_architecture == "CONV":
            model_args["kspace_conv_hidden_channels"] = kspace_conv_hidden_channels
            model_args["kspace_conv_n_convs"] = kspace_conv_n_convs
            model_args["kspace_conv_batchnorm"] = kspace_conv_batchnorm

        if imspace_model_architecture == "UNET":
            model_args["imspace_unet_num_filters"] = imspace_unet_num_filters
            model_args["imspace_unet_num_pool_layers"] = imspace_unet_num_pool_layers
            model_args["imspace_unet_dropout_probability"] = imspace_unet_dropout_probability
            model_args["imspace_unet_padding_size"] = imspace_unet_padding_size
            model_args["imspace_unet_normalize"] = imspace_unet_normalize
        elif imspace_model_architecture == "MWCNN":
            model_args["image_mwcnn_hidden_channels"] = image_mwcnn_hidden_channels
            model_args["num_scales"] = num_scales
            model_args["bias"] = bias
            model_args["batchnorm"] = batchnorm
    elif model_name == "Learned Primal-Dual Reconstruction (LPDNet)":
        model_name = "LPDNET"
        num_primal = st.number_input("Number of Primal", min_value=1, max_value=999, value=5)
        num_dual = st.number_input("Number of Dual", min_value=1, max_value=999, value=5)
        num_iter = st.number_input("Number of Iterations", min_value=1, max_value=999, value=5)
        primal_model_architecture = st.selectbox("Primal Model Architecture", ["UNET", "MWCNN"])
        primal_in_channels = st.number_input("Primal Input Channels", min_value=1, max_value=999, value=2)
        primal_out_channels = st.number_input("Primal Output Channels", min_value=1, max_value=999, value=2)
        if primal_model_architecture == "UNET":
            primal_unet_num_filters = st.number_input("Primal UNet filters", min_value=1, max_value=999, value=16)
            primal_unet_num_pool_layers = st.number_input(
                "Primal UNet pooling layers", min_value=1, max_value=999, value=2
            )
            primal_unet_dropout_probability = st.number_input(
                "Primal UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            primal_unet_padding_size = st.number_input(
                "Primal UNet padding size", min_value=0, max_value=999, value=11
            )
            primal_unet_normalize = st.selectbox("Primal UNet normalize", [True, False])
        elif primal_model_architecture == "MWCNN":
            primal_mwcnn_hidden_channels = st.number_input(
                "Primal MWCNN hidden channels", min_value=1, max_value=999, value=16
            )
            primal_mwcnn_num_scales = st.number_input("Number of Scales", min_value=1, max_value=999, value=2)
            primal_mwcnn_bias = st.selectbox("Bias", [True, False])
            primal_mwcnn_batchnorm = st.selectbox("Batchnorm", [True, False])
        dual_model_architecture = st.selectbox("Dual Model Architecture", ["UNET", "DIDN", "CONV"])
        dual_in_channels = st.number_input("Dual Input Channels", min_value=1, max_value=999, value=2)
        dual_out_channels = st.number_input("Dual Output Channels", min_value=1, max_value=999, value=2)
        if dual_model_architecture == "DIDN":
            kspace_didn_hidden_channels = st.number_input(
                "k-space DIDN hidden channels", min_value=1, max_value=999, value=64
            )
            kspace_didn_num_dubs = st.number_input("Number of Doubles", min_value=1, max_value=999, value=2)
            kspace_didn_num_convs_recon = st.number_input(
                "Number of Convolutions in Reconstruction", min_value=1, max_value=999, value=2
            )
        elif dual_model_architecture == "CONV":
            kspace_conv_hidden_channels = st.number_input(
                "k-space Convolutional hidden channels", min_value=1, max_value=999, value=64
            )
            kspace_conv_n_convs = st.number_input(
                "k-space Convolutional number of convolutions", min_value=1, max_value=999, value=2
            )
            kspace_conv_batchnorm = st.selectbox("k-space Convolutional batchnorm", [True, False])
        elif dual_model_architecture == "UNET":
            dual_unet_num_filters = st.number_input("Dual UNet filters", min_value=1, max_value=999, value=16)
            dual_unet_num_pool_layers = st.number_input(
                "Dual UNet pooling layers", min_value=1, max_value=999, value=2
            )
            dual_unet_dropout_probability = st.number_input(
                "Dual UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            dual_unet_padding_size = st.number_input("Dual UNet padding size", min_value=0, max_value=999, value=11)
            dual_unet_normalize = st.selectbox("Dual UNet normalize", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Network", [True, False])

        model_args = {
            "num_primal": num_primal,
            "num_dual": num_dual,
            "num_iter": num_iter,
            "primal_model_architecture": primal_model_architecture,
            "primal_in_channels": primal_in_channels,
            "primal_out_channels": primal_out_channels,
            "dual_model_architecture": dual_model_architecture,
            "dual_in_channels": dual_in_channels,
            "dual_out_channels": dual_out_channels,
            "use_sens_net": use_sens_net,
        }

        if primal_model_architecture == "UNET":
            model_args["primal_unet_num_filters"] = primal_unet_num_filters
            model_args["primal_unet_num_pool_layers"] = primal_unet_num_pool_layers
            model_args["primal_unet_dropout_probability"] = primal_unet_dropout_probability
            model_args["primal_unet_padding_size"] = primal_unet_padding_size
            model_args["primal_unet_normalize"] = primal_unet_normalize
        elif primal_model_architecture == "MWCNN":
            model_args["primal_mwcnn_hidden_channels"] = primal_mwcnn_hidden_channels
            model_args["primal_mwcnn_num_scales"] = primal_mwcnn_num_scales
            model_args["primal_mwcnn_bias"] = primal_mwcnn_bias
            model_args["primal_mwcnn_batchnorm"] = primal_mwcnn_batchnorm

        if dual_model_architecture == "DIDN":
            model_args["kspace_didn_hidden_channels"] = kspace_didn_hidden_channels
            model_args["kspace_didn_num_dubs"] = kspace_didn_num_dubs
            model_args["kspace_didn_num_convs_recon"] = kspace_didn_num_convs_recon
        elif dual_model_architecture == "CONV":
            model_args["kspace_conv_hidden_channels"] = kspace_conv_hidden_channels
            model_args["kspace_conv_n_convs"] = kspace_conv_n_convs
            model_args["kspace_conv_batchnorm"] = kspace_conv_batchnorm
        elif dual_model_architecture == "UNET":
            model_args["dual_unet_num_filters"] = dual_unet_num_filters
            model_args["dual_unet_num_pool_layers"] = dual_unet_num_pool_layers
            model_args["dual_unet_dropout_probability"] = dual_unet_dropout_probability
            model_args["dual_unet_padding_size"] = dual_unet_padding_size
            model_args["dual_unet_normalize"] = dual_unet_normalize
    elif model_name == "Recurrent Inference Machines (RIM)":
        model_name = "CIRIM"
        rim_only = True
        recurrent_layer = st.selectbox("RNN Type", ["GRU", "IndRNN", "MGU"])
        conv_filters = st.text_input("Convolutional Filters", value="[64, 64, 2]")
        conv_filters = [int(x) for x in conv_filters[1:-1].split(",")]
        conv_kernels = st.text_input("Convolutional Kernels", value="[5, 3, 3]")
        conv_kernels = [int(x) for x in conv_kernels[1:-1].split(",")]
        conv_dilations = st.text_input("Convolutional Dilations", value="[1, 2, 1]")
        conv_dilations = [int(x) for x in conv_dilations[1:-1].split(",")]
        conv_bias = st.text_input("Convolutional Bias", value=[True, True, False])
        recurrent_filters = st.text_input("Recurrent Filters", value="[64, 64, 0]")
        recurrent_filters = [int(x) for x in recurrent_filters[1:-1].split(",")]
        recurrent_kernels = st.text_input("Recurrent Kernels", value="[1, 1, 0]")
        recurrent_kernels = [int(x) for x in recurrent_kernels[1:-1].split(",")]
        recurrent_dilations = st.text_input("Recurrent Dilations", value="[1, 1, 0]")
        recurrent_dilations = [int(x) for x in recurrent_dilations[1:-1].split(",")]
        recurrent_bias = st.text_input("Recurrent Bias", value=[True, True, False])
        depth = st.number_input("Depth", min_value=1, max_value=999, value=2)
        time_steps = st.number_input("Time Steps", min_value=1, max_value=999, value=8)
        conv_dim = st.number_input("Convolutional Dimension", min_value=1, max_value=3, value=2)
        keep_prediction = st.selectbox("Keep all Predictions over Time Steps", [True, False])
        accumulate_predictions = st.selectbox("Accumulate Predictions over Time Steps", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "recurrent_layer": recurrent_layer,
            "conv_filters": conv_filters,
            "conv_kernels": conv_kernels,
            "conv_dilations": conv_dilations,
            "conv_bias": conv_bias,
            "recurrent_filters": recurrent_filters,
            "recurrent_kernels": recurrent_kernels,
            "recurrent_dilations": recurrent_dilations,
            "recurrent_bias": recurrent_bias,
            "depth": depth,
            "time_steps": time_steps,
            "conv_dim": conv_dim,
            "num_cascades": 1,
            "no_dc": True,
            "keep_prediction": keep_prediction,
            "accumulate_predictions": accumulate_predictions,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Recurrent Variational Network (RVN)":
        model_name = "RVN"
        in_channels = st.number_input("Input Channels", min_value=1, max_value=999, value=2)
        recurrent_hidden_channels = st.number_input("Recurrent Hidden Channels", min_value=1, max_value=999, value=64)
        recurrent_num_layers = st.number_input("Recurrent Layers", min_value=1, max_value=999, value=4)
        num_steps = st.number_input("Time Steps", min_value=1, max_value=999, value=8)
        no_parameter_sharing = st.selectbox("No Parameter Sharing", [True, False])
        learned_initializer = st.selectbox("Learned Initializer", [True, False])
        initializer_initialization = st.selectbox(
            "Initializer Initialization", ["sense", "input_image", "zero_filled"]
        )
        initializer_channels = st.text_input("Initializer Channels", value="[32, 32, 64, 64]")
        initializer_channels = [int(x) for x in initializer_channels[1:-1].split(",")]
        initializer_dilations = st.text_input("Initializer Dilations", value="[1, 1, 2, 4]")
        initializer_dilations = [int(x) for x in initializer_dilations[1:-1].split(",")]
        initializer_multiscale = st.number_input("Initializer Multiscale", min_value=1, max_value=999, value=1)
        accumulate_predictions = st.selectbox("Accumulate Predictions over Time Steps", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "in_channels": in_channels,
            "recurrent_hidden_channels": recurrent_hidden_channels,
            "recurrent_num_layers": recurrent_num_layers,
            "num_steps": num_steps,
            "no_parameter_sharing": no_parameter_sharing,
            "learned_initializer": learned_initializer,
            "initializer_initialization": initializer_initialization,
            "initializer_channels": initializer_channels,
            "initializer_dilations": initializer_dilations,
            "initializer_multiscale": initializer_multiscale,
            "accumulate_predictions": accumulate_predictions,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "UNet":
        model_name = "UNET"
        channels = st.number_input("Number of Filters", min_value=1, max_value=999, value=64)
        pooling_layers = st.number_input("Number of Pooling Layers", min_value=1, max_value=999, value=2)
        in_channels = st.number_input("Input Channels", min_value=1, max_value=999, value=2)
        out_channels = st.number_input("Output Channels", min_value=1, max_value=999, value=1)
        padding_size = st.number_input("Padding Size", min_value=0, max_value=999, value=11)
        dropout = st.number_input("Dropout Probability", min_value=0.0, max_value=1.0, value=0.0)
        normalize = st.selectbox("Normalize", [True, False])
        norm_groups = st.number_input("Normalization Groups", min_value=1, max_value=999, value=1)
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "channels": channels,
            "pooling_layers": pooling_layers,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "padding_size": padding_size,
            "dropout": dropout,
            "normalize": normalize,
            "norm_groups": norm_groups,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Variational Network (VN)":
        model_name = "VN"
        num_cascades = st.number_input("Number of Cascades", min_value=1, max_value=999, value=8)
        channels = st.number_input("Number of Filters", min_value=1, max_value=999, value=18)
        pooling_layers = st.number_input("Number of Pooling Layers", min_value=1, max_value=999, value=4)
        padding_size = st.number_input("Padding Size", min_value=0, max_value=999, value=11)
        normalize = st.selectbox("Normalize", [True, False])
        no_dc = st.selectbox("Turn off explicit Data Consistency", [False, True])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "channels": channels,
            "pooling_layers": pooling_layers,
            "padding_size": padding_size,
            "normalize": normalize,
            "no_dc": no_dc,
            "use_sens_net": use_sens_net,
        }
    elif model_name == "Variable-Splitting Net (VSNet)":
        model_name = "VSNET"
        num_cascades = st.number_input("Number of Cascades", min_value=1, max_value=999, value=10)
        imspace_model_architecture = st.selectbox("Image space Model Architecture", ["UNET", "MWCNN"])
        imspace_in_channels = st.number_input("Image space Input Channels", min_value=1, max_value=999, value=2)
        imspace_out_channels = st.number_input("Image space Output Channels", min_value=1, max_value=999, value=2)
        if imspace_model_architecture == "CONV":
            imspace_conv_hidden_channels = st.number_input(
                "Image space Convolutional hidden channels", min_value=1, max_value=999, value=64
            )
            imspace_conv_n_convs = st.number_input(
                "Image space Convolutional number of convolutions", min_value=1, max_value=999, value=2
            )
            imspace_conv_batchnorm = st.selectbox("Image space Convolutional batchnorm", [True, False])
        elif imspace_model_architecture == "MWCNN":
            image_mwcnn_hidden_channels = st.number_input(
                "Image space MWCNN hidden channels", min_value=1, max_value=999, value=64
            )
            num_scales = st.number_input("Number of Scales", min_value=1, max_value=999, value=2)
            bias = st.selectbox("Bias", [True, False])
            batchnorm = st.selectbox("Batchnorm", [True, False])
        elif imspace_model_architecture == "UNET":
            imspace_unet_num_filters = st.number_input(
                "Image space UNet filters", min_value=1, max_value=999, value=16
            )
            imspace_unet_num_pool_layers = st.number_input(
                "Image space UNet pooling layers", min_value=1, max_value=999, value=2
            )
            imspace_unet_dropout_probability = st.number_input(
                "Image space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            imspace_unet_padding_size = st.number_input(
                "Image space UNet padding size", min_value=0, max_value=999, value=11
            )
            imspace_unet_normalize = st.selectbox("Image space UNet normalize", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Net", [False, True])

        model_args = {
            "num_cascades": num_cascades,
            "imspace_model_architecture": imspace_model_architecture,
            "imspace_in_channels": imspace_in_channels,
            "imspace_out_channels": imspace_out_channels,
            "use_sens_net": use_sens_net,
        }
        if imspace_model_architecture == "CONV":
            model_args["imspace_conv_hidden_channels"] = imspace_conv_hidden_channels
            model_args["imspace_conv_n_convs"] = imspace_conv_n_convs
            model_args["imspace_conv_batchnorm"] = imspace_conv_batchnorm
        elif imspace_model_architecture == "UNET":
            model_args["imspace_unet_num_filters"] = imspace_unet_num_filters
            model_args["imspace_unet_num_pool_layers"] = imspace_unet_num_pool_layers
            model_args["imspace_unet_dropout_probability"] = imspace_unet_dropout_probability
            model_args["imspace_unet_padding_size"] = imspace_unet_padding_size
            model_args["imspace_unet_normalize"] = imspace_unet_normalize
        elif imspace_model_architecture == "MWCNN":
            model_args["image_mwcnn_hidden_channels"] = image_mwcnn_hidden_channels
            model_args["num_scales"] = num_scales
            model_args["bias"] = bias
            model_args["batchnorm"] = batchnorm
    elif model_name == "XPDNet":
        model_name = "XPDNET"
        num_primal = st.number_input("Number of Primal", min_value=1, max_value=999, value=5)
        num_dual = st.number_input("Number of Dual", min_value=1, max_value=999, value=1)
        num_iter = st.number_input("Number of Iterations", min_value=1, max_value=999, value=20)
        use_primal_only = st.selectbox("Use Primal Only", [True, False])
        kspace_model_architecture = st.selectbox("k-space Model Architecture", ["UNET", "CONV", "DIDN"])
        kspace_in_channels = st.number_input("k-space Input Channels", min_value=1, max_value=999, value=2)
        kspace_out_channels = st.number_input("k-space Output Channels", min_value=1, max_value=999, value=2)
        if kspace_model_architecture == "UNET":
            kspace_unet_num_filters = st.number_input("k-space UNet filters", min_value=1, max_value=999, value=16)
            kspace_unet_num_pool_layers = st.number_input(
                "k-space UNet pooling layers", min_value=1, max_value=999, value=2
            )
            kspace_unet_dropout_probability = st.number_input(
                "k-space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            kspace_unet_padding_size = st.number_input(
                "k-space UNet padding size", min_value=0, max_value=999, value=11
            )
            kspace_unet_normalize = st.selectbox("k-space UNet normalize", [True, False])
        elif kspace_model_architecture == "DIDN":
            dual_didn_hidden_channels = st.number_input(
                "k-space DIDN hidden channels", min_value=1, max_value=999, value=64
            )
            dual_didn_num_dubs = st.number_input("Number of Doubles", min_value=1, max_value=999, value=2)
            dual_didn_num_convs_recon = st.number_input(
                "Number of Convolutions in Reconstruction", min_value=1, max_value=999, value=2
            )
        elif kspace_model_architecture == "CONV":
            kspace_conv_hidden_channels = st.number_input(
                "k-space Convolutional hidden channels", min_value=1, max_value=999, value=64
            )
            kspace_conv_n_convs = st.number_input(
                "k-space Convolutional number of convolutions", min_value=1, max_value=999, value=2
            )
            kspace_conv_batchnorm = st.selectbox("k-space Convolutional batchnorm", [True, False])
        dual_conv_hidden_channels = st.number_input(
            "Dual Convolutional hidden channels", min_value=1, max_value=999, value=16
        )
        dual_conv_num_dubs = st.number_input(
            "Dual Convolutional number of doubles", min_value=1, max_value=999, value=2
        )
        dual_conv_batchnorm = st.selectbox("Dual Convolutional batchnorm", [True, False])
        imspace_model_architecture = st.selectbox("Image space Model Architecture", ["UNET", "MWCNN"])
        imspace_in_channels = st.number_input("Image space Input Channels", min_value=1, max_value=999, value=2)
        imspace_out_channels = st.number_input("Image space Output Channels", min_value=1, max_value=999, value=2)
        if imspace_model_architecture == "UNET":
            imspace_unet_num_filters = st.number_input(
                "Image space UNet filters", min_value=1, max_value=999, value=16
            )
            imspace_unet_num_pool_layers = st.number_input(
                "Image space UNet pooling layers", min_value=1, max_value=999, value=2
            )
            imspace_unet_dropout_probability = st.number_input(
                "Image space UNet dropout probability", min_value=0.0, max_value=1.0, value=0.0
            )
            imspace_unet_padding_size = st.number_input(
                "Image space UNet padding size", min_value=0, max_value=999, value=11
            )
            imspace_unet_normalize = st.selectbox("Image space UNet normalize", [True, False])
        elif imspace_model_architecture == "MWCNN":
            mwcnn_hidden_channels = st.number_input(
                "Image space MWCNN hidden channels", min_value=1, max_value=999, value=16
            )
            mwcnn_num_scales = st.number_input("Number of Scales", min_value=1, max_value=999, value=2)
            mwcnn_bias = st.selectbox("Bias", [True, False])
            mwcnn_batchnorm = st.selectbox("Batchnorm", [True, False])
        normalize_image = st.selectbox("Normalize Image", [True, False])
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {
            "num_iter": num_iter,
            "use_primal_only": use_primal_only,
            "kspace_model_architecture": kspace_model_architecture,
            "kspace_in_channels": kspace_in_channels,
            "kspace_out_channels": kspace_out_channels,
            "dual_conv_hidden_channels": dual_conv_hidden_channels,
            "dual_conv_num_dubs": dual_conv_num_dubs,
            "dual_conv_batchnorm": dual_conv_batchnorm,
            "imspace_model_architecture": imspace_model_architecture,
            "imspace_in_channels": imspace_in_channels,
            "imspace_out_channels": imspace_out_channels,
            "normalize_image": normalize_image,
            "use_sens_net": use_sens_net,
        }

        if kspace_model_architecture == "UNET":
            model_args["kspace_unet_num_filters"] = kspace_unet_num_filters
            model_args["kspace_unet_num_pool_layers"] = kspace_unet_num_pool_layers
            model_args["kspace_unet_dropout_probability"] = kspace_unet_dropout_probability
            model_args["kspace_unet_padding_size"] = kspace_unet_padding_size
            model_args["kspace_unet_normalize"] = kspace_unet_normalize
        elif kspace_model_architecture == "DIDN":
            model_args["dual_didn_hidden_channels"] = dual_didn_hidden_channels
            model_args["dual_didn_num_dubs"] = dual_didn_num_dubs
            model_args["dual_didn_num_convs_recon"] = dual_didn_num_convs_recon
        elif kspace_model_architecture == "CONV":
            model_args["kspace_conv_hidden_channels"] = kspace_conv_hidden_channels
            model_args["kspace_conv_n_convs"] = kspace_conv_n_convs
            model_args["kspace_conv_batchnorm"] = kspace_conv_batchnorm

        if imspace_model_architecture == "UNET":
            model_args["imspace_unet_num_filters"] = imspace_unet_num_filters
            model_args["imspace_unet_num_pool_layers"] = imspace_unet_num_pool_layers
            model_args["imspace_unet_dropout_probability"] = imspace_unet_dropout_probability
            model_args["imspace_unet_padding_size"] = imspace_unet_padding_size
            model_args["imspace_unet_normalize"] = imspace_unet_normalize
        elif imspace_model_architecture == "MWCNN":
            model_args["mwcnn_hidden_channels"] = mwcnn_hidden_channels
            model_args["mwcnn_num_scales"] = mwcnn_num_scales
            model_args["mwcnn_bias"] = mwcnn_bias
            model_args["mwcnn_batchnorm"] = mwcnn_batchnorm
    elif model_name == "Zero-Filled":
        mode = "run"
        model_name = "ZF"
        use_sens_net = st.selectbox("Use Sensitivity Network", [False, True])

        model_args = {"use_sens_net": use_sens_net}
    elif model_name == "Attention UNet (2D)":
        model_name = "SEGMENTATIONATTENTIONUNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "AttentionUNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Dynamic UNet (2D)":
        model_name = "DYNUNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.text_input("Segmentation Module Channels", value="[64, 128, 256, 512]")
        segmentation_module_kernel_size = st.text_input("Segmentation Kernel Size", value="[3, 3, 3, 3]")
        segmentation_module_strides = st.text_input("Segmentation Kernel Size", value="[1, 1, 1, 1]")
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        segmentation_module_norm = st.selectbox("Segmentation Module Norm", ["instance"])
        segmentation_module_activation = st.selectbox("Segmentation Module Activation", ["leakyrelu"])
        segmentation_module_deep_supervision = st.selectbox("Segmentation Module Deep Supervision", [True, False])
        segmentation_module_deep_supervision_levels = st.number_input(
            "Segmentation Module Deep Supervision Levels", min_value=1, max_value=999, value=2
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "DYNUNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_kernel_size": segmentation_module_kernel_size,
            "segmentation_module_strides": segmentation_module_strides,
            "segmentation_module_dropout": segmentation_module_dropout,
            "segmentation_module_norm": segmentation_module_norm,
            "segmentation_module_activation": segmentation_module_activation,
            "segmentation_module_deep_supervision": segmentation_module_deep_supervision,
            "segmentation_module_deep_supervision_levels": segmentation_module_deep_supervision_levels,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Lambda UNet (2D)":
        model_name = "SEGMENTATIONLAMBDAUNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        segmentation_module_query_depth = st.number_input(
            "Segmentation Module Query Depth", min_value=1, max_value=999, value=16
        )
        segmentation_module_receptive_kernel_size = st.number_input(
            "Segmentation Module Receptive Kernel Size", min_value=1, max_value=999, value=3
        )
        segmentation_module_temporal_kernel = st.number_input(
            "Segmentation Module Temporal Kernel", min_value=1, max_value=999, value=3
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "LambdaUNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_dropout": segmentation_module_dropout,
            "segmentation_module_query_depth": segmentation_module_query_depth,
            "segmentation_module_receptive_kernel_size": segmentation_module_receptive_kernel_size,
            "segmentation_module_temporal_kernel": segmentation_module_temporal_kernel,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "Lambda UNet (3D)":
        segmentation_dim = "3D"
        model_name = "SEGMENTATIONLAMBDAUNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        segmentation_module_query_depth = st.number_input(
            "Segmentation Module Query Depth", min_value=1, max_value=999, value=16
        )
        segmentation_module_receptive_kernel_size = st.number_input(
            "Segmentation Module Receptive Kernel Size", min_value=1, max_value=999, value=3
        )
        segmentation_module_temporal_kernel = st.number_input(
            "Segmentation Module Temporal Kernel", min_value=1, max_value=999, value=3
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "LambdaUNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_dropout": segmentation_module_dropout,
            "segmentation_module_query_depth": segmentation_module_query_depth,
            "segmentation_module_receptive_kernel_size": segmentation_module_receptive_kernel_size,
            "segmentation_module_temporal_kernel": segmentation_module_temporal_kernel,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "UNet (2D)":
        model_name = "SEGMENTATIONUNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "UNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "UNet (3D)":
        segmentation_dim = "3D"
        model_name = "SEGMENTATION3DUNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_pooling_layers = st.number_input(
            "Segmentation Module Pooling Layers", min_value=1, max_value=999, value=2
        )
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "UNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_pooling_layers": segmentation_module_pooling_layers,
            "segmentation_module_dropout": segmentation_module_dropout,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "UNetR (2D)":
        model_name = "SEGMENTATIONUNETR"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_img_size = st.text_input("Segmentation Module Image Size", value="[256, 256]")
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_hidden_size = st.number_input(
            "Segmentation Module Hidden Size", min_value=1, max_value=9999, value=768
        )
        segmentation_module_mlp_dim = st.number_input(
            "Segmentation Module MLP Dim", min_value=1, max_value=9999, value=3072
        )
        segmentation_module_num_heads = st.number_input(
            "Segmentation Module Num Heads", min_value=1, max_value=999, value=12
        )
        segmentation_module_pos_embed = st.selectbox("Segmentation Module Positional Embedding", ["conv"])
        segmentation_module_norm_name = st.selectbox("Segmentation Module Normalization", ["instance"])
        segmentation_module_conv_block = st.selectbox("Segmentation Module Convolution Block", [True, False])
        segmentation_module_res_block = st.selectbox("Segmentation Module Residual Block", [True, False])
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        segmentation_module_qkv_bias = st.selectbox("Segmentation Module QKV Bias", [False, True])
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "UNETR",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_hidden_size": segmentation_module_hidden_size,
            "segmentation_module_mlp_dim": segmentation_module_mlp_dim,
            "segmentation_module_num_heads": segmentation_module_num_heads,
            "segmentation_module_pos_embed": segmentation_module_pos_embed,
            "segmentation_module_norm_name": segmentation_module_norm_name,
            "segmentation_module_conv_block": segmentation_module_conv_block,
            "segmentation_module_res_block": segmentation_module_res_block,
            "segmentation_module_dropout": segmentation_module_dropout,
            "segmentation_module_qkv_bias": segmentation_module_qkv_bias,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }
    elif model_name == "VNet (2D)":
        model_name = "SEGMENTATIONVNET"
        segmentation_module_input_channels = st.number_input(
            "Segmentation Module Input Channels", min_value=1, max_value=999, value=1
        )
        segmentation_module_output_channels = st.number_input(
            "Segmentation Module Output Channels", min_value=1, max_value=999, value=2
        )
        segmentation_module_channels = st.number_input(
            "Segmentation Module Channels", min_value=1, max_value=999, value=64
        )
        segmentation_module_activation = st.selectbox("Segmentation Module Activation", ["elu"])
        segmentation_module_dropout = st.number_input(
            "Segmentation Module Dropout", min_value=0.0, max_value=1.0, value=0.0
        )
        segmentation_module_bias = st.selectbox("Segmentation Module Bias", [False, True])
        segmentation_module_padding_size = st.number_input(
            "Segmentation Module Padding Size", min_value=1, max_value=999, value=15
        )
        magnitude_input = st.selectbox("Magnitude Input", [True, False])
        normalize_segmentation_output = st.selectbox("Normalize Segmentation Output", [True, False])

        model_args = {
            "use_reconstruction_module": False,
            "segmentation_module": "VNet",
            "segmentation_module_input_channels": segmentation_module_input_channels,
            "segmentation_module_output_channels": segmentation_module_output_channels,
            "segmentation_module_channels": segmentation_module_channels,
            "segmentation_module_activation": segmentation_module_activation,
            "segmentation_module_dropout": segmentation_module_dropout,
            "segmentation_module_bias": segmentation_module_bias,
            "segmentation_module_padding_size": segmentation_module_padding_size,
            "magnitude_input": magnitude_input,
            "normalize_segmentation_output": normalize_segmentation_output,
        }

with st.expander("Dataset", expanded=False):
    dataset_type = st.selectbox("Dataset Type", ["Custom", "CC359", "fastMRI"])
    dataset_type = dataset_type.lower()

    if mode == "train":
        # train_data_path = st.text_input("Train Data Path", value="/data/train")
        train_data_path = st.text_input(
            "Train Data Path", value="/data/projects/recon/data/public/3D_FSE_PD_Knees/h5/multicoil_train"
        )
        train_coil_sensitivity_maps_path = st.text_input("Train Coil Sensitivity Maps Path", value=None)
        train_mask_path = st.text_input("Train Mask Path", value=None)
        # val_data_path = st.text_input("Validation Data Path", value="/data/val")
        val_data_path = st.text_input(
            "Validation Data Path", value="/data/projects/recon/data/public/3D_FSE_PD_Knees/h5/multicoil_val"
        )
        val_coil_sensitivity_maps_path = st.text_input("Validation Coil Sensitivity Maps Path", value=None)
        val_mask_path = st.text_input("Validation Mask Path", value=None)
        if task in ["segmentation", "multitask"]:
            train_segmentations_path = st.text_input("Train Segmentations Path", value=None)
            train_initial_predictions_path = st.text_input("Train Initial Predictions Path", value=None)
            val_segmentations_path = st.text_input("Validation Segmentations Path", value=None)
            val_initial_predictions_path = st.text_input("Validation Initial Predictions Path", value=None)
    else:
        # test_data_path = st.text_input("Test Data Path", value="/data/test")
        test_data_path = st.text_input(
            "Test Data Path", value="/data/projects/recon/data/public/3D_FSE_PD_Knees/h5/multicoil_val"
        )
        test_coil_sensitivity_maps_path = st.text_input("Test Coil Sensitivity Maps Path", value=None)
        test_mask_path = st.text_input("Test Mask Path", value=None)
        if task in ["segmentation", "multitask"]:
            test_segmentations_path = st.text_input("Test Segmentations Path", value=None)
            test_initial_predictions_path = st.text_input("Test Initial Predictions Path", value=None)

with st.expander("Data Dimensionality", expanded=False):
    dimensionality = st.selectbox("Dimensionality", [2, 3])
    consecutive_slices = st.number_input("Consecutive Slices", min_value=1, max_value=999, value=1)
    slice_dim = st.number_input("Slice Dimension", min_value=0, max_value=4, value=0)
    first_spatial_dim = st.number_input("First Spatial Dimension", min_value=-4, max_value=-1, value=-2)
    second_spatial_dim = st.number_input("Second Spatial Dimension", min_value=-4, max_value=-1, value=-1)
    coil_dim = st.number_input("Coil Dimension", min_value=0, max_value=4, value=1)

with st.expander("Preprocessing Transforms", expanded=False):
    coil_combination_method = st.selectbox("Coil Combination Method", ["SENSE", "RSS"])

    apply_prewhitening = st.selectbox("Coil Prewhitening", [False, True])
    if apply_prewhitening:
        prewhitening_scale_factor = st.number_input("Prewhitening Scale Factor", value=1.0)
        prewhitening_patch_start = st.number_input("Prewhitening Patch Start", value=10)
        prewhitening_patch_length = st.number_input("Prewhitening Patch Length", value=30)
    else:
        prewhitening_scale_factor = None
        prewhitening_patch_start = None
        prewhitening_patch_length = None

    apply_gcc = st.selectbox("Geometric Decomposition Coil Compression", [False, True])
    if apply_gcc:
        gcc_virtual_coils = st.number_input("Virtual Coils", value=4)
        gcc_calib_lines = st.number_input("Calibration Lines", value=24)
        gcc_align_data = st.selectbox("Align Data", [True, False])
    else:
        gcc_virtual_coils = None
        gcc_calib_lines = None
        gcc_align_data = None

    half_scan_percentage = st.number_input("Half Scan Percentage", min_value=0.0, max_value=1.0, value=0.0)
    crop_size = st.text_input("Crop Size", value="None")
    kspace_crop = st.selectbox("Crop in k-space", [False, True])
    crop_before_masking = st.selectbox("Crop before masking", [True, False])
    kspace_zero_filling_size = st.text_input("k-space zero filling size", value="None")
    normalize_inputs = st.selectbox("Normalize Inputs", [True, False])
    max_normalization_value = st.selectbox("Normalize by the maximum value", [True, False])

with st.expander("Fast Fourier Transform Parameters", expanded=False):
    forward_operator = st.selectbox("Forward Operator", ["FFT2"])
    adjoint_operator = st.selectbox("Backward Operator", ["IFFT2"])
    fft_centered = st.selectbox("Centered", [False, True])
    fft_normalization = st.selectbox("Normalization", ["backward", "forward", "orthogonal", "none"])
    if fft_normalization == "none":
        fft_normalization = None
    elif fft_normalization == "orthogonal":
        fft_normalization = "ortho"

with st.expander("Retrospective Subsampling", expanded=False):
    # subsampling = st.selectbox(" ", [False, True])
    subsampling = st.selectbox(" ", [True, False])
    # if true, open the subsampling parameters
    if subsampling:
        subsampling_type = st.selectbox(
            "Subsampling Type", ["equispaced1d", "equispaced2d", "gaussian1d", "gaussian2d", "poisson2d", "random1d"]
        )
        # allow to enter multiple acceleration factors as list
        subsampling_accelerations = st.text_input("Accelerations", value="[4, 8]")
        subsampling_accelerations = [int(x) for x in subsampling_accelerations.strip("[]").split(",")]
        subsampling_center_fractions = st.text_input("Center Fractions", value="[0.08, 0.04]")
        subsampling_center_fractions = [float(x) for x in subsampling_center_fractions.strip("[]").split(",")]
        subsampling_center_scale = st.number_input("Center Scale", min_value=0.0, max_value=1.0, value=0.02)
        subsampling_shift = st.selectbox("Shift subsampling mask", [False, True])
        subsampling_remask = st.selectbox("Remask", [False, True])
    else:
        subsampling_type = None
        subsampling_accelerations = None
        subsampling_center_fractions = None
        subsampling_center_scale = None
        subsampling_shift = None
        subsampling_remask = None

with st.expander("Loss function", expanded=False):
    if task == "reconstruction":
        train_loss_fn = st.selectbox("Train Loss Function", ["l1", "mse", "ssim"])
        val_loss_fn = st.selectbox("Validation Loss Function", ["l1", "mse", "ssim"])
    elif task == "quantitative":
        loss_fn = st.selectbox("Loss Function", ["ssim", "l1", "mse"])
        if loss_fn == "ssim":
            loss_regularization_factors_R2star = st.number_input(
                "Loss Regularization Factor R2star", min_value=0.0, max_value=100000.0, value=3.0
            )
            loss_regularization_factors_S0 = st.number_input(
                "Loss Regularization Factor S0", min_value=0.0, max_value=100000.0, value=1.0
            )
            loss_regularization_factors_B0 = st.number_input(
                "Loss Regularization Factor B0", min_value=0.0, max_value=100000.0, value=1.0
            )
            loss_regularization_factors_phi = st.number_input(
                "Loss Regularization Factor phi", min_value=0.0, max_value=100000.0, value=1.0
            )
        else:
            loss_regularization_factors_R2star = st.number_input(
                "Loss Regularization Factor R2star", min_value=0.0, max_value=100000.0, value=300.0
            )
            loss_regularization_factors_S0 = st.number_input(
                "Loss Regularization Factor S0", min_value=0.0, max_value=100000.0, value=500.0
            )
            loss_regularization_factors_B0 = st.number_input(
                "Loss Regularization Factor B0", min_value=0.0, max_value=100000.0, value=20000.0
            )
            loss_regularization_factors_phi = st.number_input(
                "Loss Regularization Factor phi", min_value=0.0, max_value=100000.0, value=500.0
            )
    elif task in ["segmentation", "multitask"]:
        segmentation_loss_fn = st.selectbox(
            "Segmentation Loss Function", ["dice+cross_entropy", "cross_entropy", "dice"]
        )
        if segmentation_loss_fn in ["cross_entropy", "dice+cross_entropy"]:
            cross_entropy_loss_num_samples = st.number_input(
                "Cross Entropy Loss Number of Samples", min_value=0, max_value=999, value=1
            )
            cross_entropy_loss_ignore_index = st.number_input(
                "Cross Entropy Loss Ignore Index", min_value=-999, max_value=999, value=-100
            )
            cross_entropy_loss_reduction = st.selectbox("Cross Entropy Loss Reduction", ["mean", "sum", "none"])
            cross_entropy_loss_label_smoothing = st.number_input(
                "Cross Entropy Loss Label Smoothing", min_value=0.0, max_value=999.0, value=0.0
            )
            cross_entropy_loss_weight = st.text_input(
                "Cross Entropy Loss Weight. Length = number of classes ", value="[0.5, 0.5]"
            )
        if segmentation_loss_fn in ["dice", "dice+cross_entropy"]:
            dice_loss_include_background = st.selectbox("Dice Loss Include Background", [True, False])
            dice_loss_to_onehot_y = st.selectbox("Dice Loss To Onehot Y", [False, True])
            dice_loss_sigmoid = st.selectbox("Dice Loss Sigmoid", [False, True])
            dice_loss_softmax = st.selectbox("Dice Loss Softmax", [False, True])
            dice_loss_other_act = st.selectbox("Dice Loss Other Act", [None, "tanh"])
            dice_loss_squared_pred = st.selectbox("Dice Loss Squared Pred", [False, True])
            dice_loss_jaccard = st.selectbox("Dice Loss Jaccard", [False, True])
            dice_loss_flatten = st.selectbox("Dice Loss Flatten", [False, True])
            dice_loss_reduction = st.selectbox("Dice Loss Reduction", ["mean", "sum", "none"])
            dice_loss_smooth_nr = st.number_input("Dice Loss Smooth Nr", min_value=0.0, max_value=1.0, value=1e-5)
            dice_loss_smooth_dr = st.number_input("Dice Loss Smooth Dr", min_value=0.0, max_value=1.0, value=1e-5)
            dice_loss_batch = st.selectbox("Dice Loss Batch", [True, False])
        if segmentation_loss_fn == "dice+cross_entropy":
            cross_entropy_loss_weighting_factor = st.number_input(
                "Cross Entropy Loss Weighting Factor", min_value=0.0, max_value=1.0, value=0.5
            )
            dice_loss_weighting_factor = st.number_input(
                "Dice Loss Weighting Factor", min_value=0.0, max_value=1.0, value=0.5
            )
        elif segmentation_loss_fn == "cross_entropy":
            cross_entropy_loss_weighting_factor = 1.0
        elif segmentation_loss_fn == "dice":
            dice_loss_weighting_factor = 1.0

if task in ["segmentation", "multitask"]:
    with st.expander("Segmentation Metrics", expanded=False):
        cross_entropy_metric_num_samples = st.number_input(
            "Cross Entropy Metric Number of Samples", min_value=0, max_value=999, value=1
        )
        cross_entropy_metric_ignore_index = st.number_input(
            "Cross Entropy Metric Ignore Index", min_value=-999, max_value=999, value=-100
        )
        cross_entropy_metric_reduction = st.selectbox("Cross Entropy Metric Reduction", ["mean", "sum", "none"])
        cross_entropy_metric_label_smoothing = st.number_input(
            "Cross Entropy Metric Label Smoothing", min_value=0.0, max_value=1.0, value=0.0
        )
        cross_entropy_metric_weight = st.text_input(
            "Cross Entropy Metric Weight. Length = number of classes ", value="[0.5, 0.5]"
        )
        dice_metric_include_background = st.selectbox("Dice Metric Include Background", [True, False])
        dice_metric_to_onehot_y = st.selectbox("Dice Metric To Onehot Y", [False, True])
        dice_metric_sigmoid = st.selectbox("Dice Metric Sigmoid", [False, True])
        dice_metric_softmax = st.selectbox("Dice Metric Softmax", [False, True])
        dice_metric_other_act = st.selectbox("Dice Metric Other Act", [None, "tanh"])
        dice_metric_squared_pred = st.selectbox("Dice Metric Squared Pred", [False, True])
        dice_metric_jaccard = st.selectbox("Dice Metric Jaccard", [False, True])
        dice_metric_flatten = st.selectbox("Dice Metric Flatten", [False, True])
        dice_metric_reduction = st.selectbox("Dice Metric Reduction", ["mean", "sum", "none"])
        dice_metric_smooth_nr = st.number_input("Dice Metric Smooth Nr", min_value=0.0, max_value=1.0, value=1e-5)
        dice_metric_smooth_dr = st.number_input("Dice Metric Smooth Dr", min_value=0.0, max_value=1.0, value=1e-5)
        dice_metric_batch = st.selectbox("Dice Metric Batch", [True, False])

# data loader options
with st.expander("Data Loader", expanded=False):
    # sample_rate = st.number_input("Sample Rate", min_value=0.0, max_value=1.0, value=1.0)
    sample_rate = st.number_input("Sample Rate", min_value=0.0, max_value=1.0, value=0.001)
    volume_sample_rate = st.text_input("Volume Sample Rate", value="None")
    use_dataset_cache = st.selectbox("Use Dataset Cache", [False, True])
    dataset_cache_file = st.text_input("Dataset Cache File", value="None")
    num_cols = st.number_input("Number of Columns", min_value=0, max_value=999, value=0)
    if num_cols == 0:
        num_cols = None
    data_saved_per_slice = st.selectbox("Data Saved Per Slice", [False, True])
    batch_size = st.number_input("Batch Size", min_value=1, max_value=999, value=1)
    num_workers = st.number_input("Number of Workers", min_value=1, max_value=999, value=4)
    shuffle = st.selectbox("Shuffle", [True, False])
    pin_memory = st.selectbox("Pin Memory", [False, True])
    drop_last = st.selectbox("Drop Last", [False, True])

# optimizer options
with st.expander("Optimizer", expanded=False):
    optimizers = ["sgd", "adam", "adamw", "adadelta", "adamax", "adagrad", "rmsprop", "rprop", "novograd", "adafactor"]
    # order alphabetically and capitalize first letter
    optimizers = sorted(optimizers, key=lambda x: x.lower())
    optimizers = [x.capitalize() for x in optimizers]
    # bring Adam to first position
    optimizers = [optimizers.pop(optimizers.index("Adam"))] + optimizers
    optimizer = st.selectbox("Optimizer", optimizers)

    learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, value=0.001)
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1.0, value=0.0)
    betas = st.text_input("Betas", value="(0.9, 0.98)")
    betas = [float(x) for x in betas.strip("()").split(",")]
    schedulers = [
        "CosineAnnealing",
        "CyclicLR",
        "ExponentialLR",
        "InverseSquareRootAnnealing",
        "NoamAnnealing",
        "PolynomialDecayAnnealing",
        "PolynomialHoldDecayAnnealing",
        "ReduceLROnPlateau",
        "StepLR",
        "SquareAnnealing",
        "SquareRootAnnealing",
        "T5InverseSquareRootAnnealing",
        "WarmupAnnealing",
        "WarmupHoldPolicy",
        "WarmupPolicy",
    ]
    # bring InverseSquareRootAnnealing to first position
    schedulers = [schedulers.pop(schedulers.index("InverseSquareRootAnnealing"))] + schedulers
    scheduler = st.selectbox("Scheduler", schedulers)

    scheduler_min_lr = st.number_input("Scheduler Min LR", min_value=0.0, max_value=1.0, value=0.0)
    scheduler_last_epoch = st.number_input("Scheduler Last Epoch", min_value=-1, max_value=999, value=-1)
    scheduler_warmup_ratio = st.number_input("Scheduler Warmup Ratio", min_value=0.0, max_value=1.0, value=0.1)

# pytorch lightning trainer options
with st.expander("Trainer", expanded=False):
    trainer_strategies = [
        "bagua",
        "colossalai",
        "ddp",
        "ddp_find_unused_parameters_false",
        "ddp_fork",
        "ddp_fork_find_unused_parameters_false",
        "ddp_fully_sharded",
        "ddp_notebook",
        "ddp_notebook_find_unused_parameters_false",
        "ddp_sharded",
        "ddp_sharded_find_unused_parameters_false",
        "ddp_sharded_spawn",
        "ddp_sharded_spawn_find_unused_parameters_false",
        "ddp_spawn",
        "ddp_spawn_find_unused_parameters_false",
        "deepspeed",
        "deepspeed_stage_1",
        "deepspeed_stage_2",
        "deepspeed_stage_2_offload",
        "deepspeed_stage_3",
        "deepspeed_stage_3_offload",
        "deepspeed_stage_3_offload_nvme",
        "dp",
        "fsdp",
        "fsdp_native",
        "fsdp_native_full_shard_offload",
        "horovod",
        "hpu_parallel",
        "hpu_single",
        "ipu_strategy",
        "single_device",
        "single_tpu",
        "tpu_spawn",
        "tpu_spawn_debug",
    ]
    # order alphabetically and capitalize first letter
    trainer_strategies = sorted(trainer_strategies, key=lambda x: x.lower())
    # bring InverseSquareRootAnnealing to first position
    trainer_strategies = [trainer_strategies.pop(trainer_strategies.index("ddp"))] + trainer_strategies
    trainer_strategy = st.selectbox("Strategy", trainer_strategies)

    trainer_accelerator = st.selectbox("Accelerator", ["gpu", "cpu"])
    trainer_devices = st.number_input("Devices", min_value=1, max_value=999, value=1)
    trainer_num_nodes = st.number_input("Number of Nodes", min_value=1, max_value=999, value=1)
    trainer_precision = st.selectbox("Precision", [16, 32])
    trainer_max_epochs = st.number_input("Max Epochs", min_value=1, max_value=999, value=200)
    trainer_enable_checkpointing = st.selectbox("Enable Checkpointing", [False, True])
    trainer_logger = st.selectbox("Logger", [False, True])
    trainer_log_every_n_steps = st.number_input("Log Every N Steps", min_value=1, max_value=999, value=50)
    trainer_check_val_every_n_epoch = st.number_input("Check Val Every N Epoch", min_value=-1, max_value=999, value=-1)
    trainer_max_steps = st.number_input("Max Steps", min_value=-1, max_value=999, value=-1)

# export options
with st.expander("Export", expanded=False):
    # export_path = st.text_input("Export Path", value=".")
    export_path = st.text_input("Export Path", value="/data/projects/recon/other/dkarkalousos/delete")
    Path(export_path).mkdir(parents=True, exist_ok=True)
    create_tensorboard_logger = st.selectbox("Create Tensorboard Logger", [True, False])
    create_wandb_logger = st.selectbox("Create Wandb Logger", [False, True])
    wandb_logger_kwargs = st.text_input("Wandb Logger Details", value=".")
    log_images = st.selectbox("Log images on tensorboard or wandb", [True, False])
    ema = st.selectbox("Enable Exponential Moving Average", [False, True])

# replace values in keys in base_config with values in model_args
if model_name == "CRNNET":
    model_name = "crnn"
elif model_name == "CASCADENET":
    model_name = "ccnn"
elif model_name == "CIRIM" and rim_only:
    model_name = "RIM"
elif model_name == "qCIRIM" and rim_only:
    model_name = "qRIM"
elif "SEGMENTATION" in model_name:
    # remove "SEGMENTATION" from model_name
    model_name = model_name.replace("SEGMENTATION", "")

if model_name == "LAMBDAUNET" and segmentation_dim == "2D":
    model_name = "LAMBDAUNET2D"
elif model_name == "LAMBDAUNET" and segmentation_dim == "3D":
    model_name = "LAMBDAUNET3D"

if model_name == "3DUNET":
    model_name = "unet3d"

if model_name == "UNET" and segmentation_dim == "2D" and task in ["multitask", "segmentation"]:
    model_name = "unet2d"

if model_name == "MTLRS" and model_args["task_adaption_type"] is None:
    model_name = "JRS"

base_config = {
    "pretrained": pretrained,
    "checkpoint": checkpoint_path,
    "mode": mode.lower(),
    "model": {},
    "trainer": {},
    "exp_manager": {},
}
base_config["model"]["spatial_dims"] = [-2, -1]
if mode == "train":
    base_config["model"]["train_ds"] = {}
    base_config["model"]["train_ds"]["mask_args"] = {}
    base_config["model"]["train_ds"]["spatial_dims"] = [-2, -1]
    base_config["model"]["validation_ds"] = {}
    base_config["model"]["validation_ds"]["mask_args"] = {}
    base_config["model"]["validation_ds"]["spatial_dims"] = [-2, -1]
else:
    base_config["model"]["test_ds"] = {}
    base_config["model"]["test_ds"]["mask_args"] = {}
    base_config["model"]["test_ds"]["spatial_dims"] = [-2, -1]
base_config["model"]["optim"] = {}
base_config["model"]["optim"]["sched"] = {}

base_config["model"]["model_name"] = model_name
for key, value in model_args.items():
    if not _isnone_isnull_(value):
        if task == "quantitative":
            base_config["model"]["loss_regularization_factors"] = {}
            if key == "loss_regularization_factors_R2star":
                base_config["model"]["loss_regularization_factors"]["R2star"] = value
            elif key == "loss_regularization_factors_S0":
                base_config["model"]["loss_regularization_factors"]["S0"] = value
            elif key == "loss_regularization_factors_B0":
                base_config["model"]["loss_regularization_factors"]["B0"] = value
            elif key == "loss_regularization_factors_phi":
                base_config["model"]["loss_regularization_factors"]["phi"] = value
        else:
            base_config["model"][key] = value

base_config["model"]["dimensionality"] = dimensionality
if task == "reconstruction":
    base_config["model"]["train_loss_fn"] = train_loss_fn.lower()
    base_config["model"]["val_loss_fn"] = val_loss_fn.lower()
elif task == "quantitative":
    base_config["model"]["loss_fn"] = loss_fn.lower()
    base_config["model"]["loss_regularization_factors"]["R2star"] = (
        loss_regularization_factors_R2star
        if not _isnone_isnull_(loss_regularization_factors_R2star)
        else base_config["model"]["loss_regularization_factors"]["R2star"]
    )
    base_config["model"]["loss_regularization_factors"]["S0"] = (
        loss_regularization_factors_S0
        if not _isnone_isnull_(loss_regularization_factors_S0)
        else base_config["model"]["loss_regularization_factors"]["S0"]
    )
    base_config["model"]["loss_regularization_factors"]["B0"] = (
        loss_regularization_factors_B0
        if not _isnone_isnull_(loss_regularization_factors_B0)
        else base_config["model"]["loss_regularization_factors"]["B0"]
    )
    base_config["model"]["loss_regularization_factors"]["phi"] = (
        loss_regularization_factors_phi
        if not _isnone_isnull_(loss_regularization_factors_phi)
        else base_config["model"]["loss_regularization_factors"]["phi"]
    )
elif task in ["segmentation", "multitask"]:
    base_config["model"]["complex_data"] = complex_data
    base_config["model"]["segmentation_loss_fn"] = segmentation_loss_fn
    if segmentation_loss_fn in ["cross_entropy", "dice+cross_entropy"]:
        base_config["model"]["cross_entropy_loss_num_samples"] = cross_entropy_loss_num_samples
        base_config["model"]["cross_entropy_loss_ignore_index"] = cross_entropy_loss_ignore_index
        base_config["model"]["cross_entropy_loss_reduction"] = cross_entropy_loss_reduction
        base_config["model"]["cross_entropy_loss_label_smoothing"] = cross_entropy_loss_label_smoothing
        base_config["model"]["cross_entropy_loss_weight"] = cross_entropy_loss_weight
    if segmentation_loss_fn in ["dice", "dice+cross_entropy"]:
        base_config["model"]["dice_loss_include_background"] = dice_loss_include_background
        base_config["model"]["dice_loss_to_onehot_y"] = dice_loss_to_onehot_y
        base_config["model"]["dice_loss_sigmoid"] = dice_loss_sigmoid
        base_config["model"]["dice_loss_softmax"] = dice_loss_softmax
        base_config["model"]["dice_loss_other_act"] = dice_loss_other_act
        base_config["model"]["dice_loss_squared_pred"] = dice_loss_squared_pred
        base_config["model"]["dice_loss_jaccard"] = dice_loss_jaccard
        base_config["model"]["dice_loss_flatten"] = dice_loss_flatten
        base_config["model"]["dice_loss_reduction"] = dice_loss_reduction
        base_config["model"]["dice_loss_smooth_nr"] = dice_loss_smooth_nr
        base_config["model"]["dice_loss_smooth_dr"] = dice_loss_smooth_dr
        base_config["model"]["dice_loss_batch"] = dice_loss_batch

    if segmentation_loss_fn in ["dice", "dice+cross_entropy"]:
        base_config["model"]["cross_entropy_loss_weighting_factor"] = cross_entropy_loss_weighting_factor
        base_config["model"]["dice_loss_weighting_factor"] = dice_loss_weighting_factor
    elif segmentation_loss_fn == "cross_entropy":
        base_config["model"]["cross_entropy_loss_weighting_factor"] = cross_entropy_loss_weighting_factor
    elif segmentation_loss_fn == "dice":
        base_config["model"]["dice_loss_weighting_factor"] = dice_loss_weighting_factor

    base_config["model"]["cross_entropy_metric_num_samples"] = cross_entropy_metric_num_samples
    base_config["model"]["cross_entropy_metric_ignore_index"] = cross_entropy_metric_ignore_index
    base_config["model"]["cross_entropy_metric_reduction"] = cross_entropy_metric_reduction
    base_config["model"]["cross_entropy_metric_label_smoothing"] = cross_entropy_metric_label_smoothing
    base_config["model"]["cross_entropy_metric_weight"] = cross_entropy_metric_weight
    base_config["model"]["dice_metric_include_background"] = dice_metric_include_background
    base_config["model"]["dice_metric_to_onehot_y"] = dice_metric_to_onehot_y
    base_config["model"]["dice_metric_sigmoid"] = dice_metric_sigmoid
    base_config["model"]["dice_metric_softmax"] = dice_metric_softmax
    base_config["model"]["dice_metric_other_act"] = dice_metric_other_act
    base_config["model"]["dice_metric_squared_pred"] = dice_metric_squared_pred
    base_config["model"]["dice_metric_jaccard"] = dice_metric_jaccard
    base_config["model"]["dice_metric_flatten"] = dice_metric_flatten
    base_config["model"]["dice_metric_reduction"] = dice_metric_reduction
    base_config["model"]["dice_metric_smooth_nr"] = dice_metric_smooth_nr
    base_config["model"]["dice_metric_smooth_dr"] = dice_metric_smooth_dr
    base_config["model"]["dice_metric_batch"] = dice_metric_batch

if model_name in ["IDSLR", "IDSLRUNET"]:
    base_config["model"]["coil_combination_method"] = "RSS"
else:
    base_config["model"]["coil_combination_method"] = coil_combination_method

base_config["model"]["fft_centered"] = fft_centered
base_config["model"]["fft_normalization"] = fft_normalization
base_config["model"]["spatial_dims"][0] = first_spatial_dim
base_config["model"]["spatial_dims"][1] = second_spatial_dim
base_config["model"]["coil_dim"] = coil_dim
base_config["model"]["consecutive_slices"] = consecutive_slices
base_config["model"]["log_images"] = log_images

if mode == "train":
    base_config["model"]["train_ds"]["data_path"] = train_data_path
    base_config["model"]["train_ds"]["coil_sensitivity_maps_path"] = train_coil_sensitivity_maps_path
    base_config["model"]["train_ds"]["mask_path"] = train_mask_path
    base_config["model"]["train_ds"]["dataset_format"] = dataset_type
    base_config["model"]["train_ds"]["sample_rate"] = sample_rate
    base_config["model"]["train_ds"]["volume_sample_rate"] = volume_sample_rate
    base_config["model"]["train_ds"]["use_dataset_cache"] = use_dataset_cache
    base_config["model"]["train_ds"]["dataset_cache_file"] = dataset_cache_file
    base_config["model"]["train_ds"]["num_cols"] = num_cols
    base_config["model"]["train_ds"]["consecutive_slices"] = consecutive_slices
    base_config["model"]["train_ds"]["data_saved_per_slice"] = data_saved_per_slice
    base_config["model"]["train_ds"]["apply_prewhitening"] = apply_prewhitening
    base_config["model"]["train_ds"]["prewhitening_scale_factor"] = prewhitening_scale_factor
    base_config["model"]["train_ds"]["prewhitening_patch_start"] = prewhitening_patch_start
    base_config["model"]["train_ds"]["prewhitening_patch_length"] = prewhitening_patch_length
    base_config["model"]["train_ds"]["apply_gcc"] = apply_gcc
    base_config["model"]["train_ds"]["gcc_virtual_coils"] = gcc_virtual_coils
    base_config["model"]["train_ds"]["gcc_calib_lines"] = gcc_calib_lines
    base_config["model"]["train_ds"]["gcc_align_data"] = gcc_align_data
    base_config["model"]["train_ds"]["coil_combination_method"] = coil_combination_method
    base_config["model"]["train_ds"]["dimensionality"] = dimensionality
    base_config["model"]["train_ds"]["mask_args"] = {}
    base_config["model"]["train_ds"]["mask_args"]["type"] = subsampling_type
    if subsampling_type is not None:
        base_config["model"]["train_ds"]["mask_args"]["accelerations"] = {}
        base_config["model"]["train_ds"]["mask_args"]["accelerations"] = [x for x in subsampling_accelerations]
        base_config["model"]["train_ds"]["mask_args"]["center_fractions"] = {}
        base_config["model"]["train_ds"]["mask_args"]["center_fractions"] = [x for x in subsampling_center_fractions]
        base_config["model"]["train_ds"]["mask_args"]["scale"] = subsampling_center_scale
    base_config["model"]["train_ds"]["mask_args"]["shift_mask"] = subsampling_shift
    base_config["model"]["train_ds"]["mask_args"]["use_seed"] = True
    base_config["model"]["train_ds"]["half_scan_percentage"] = half_scan_percentage
    base_config["model"]["train_ds"]["remask"] = subsampling_remask
    base_config["model"]["train_ds"]["crop_size"] = crop_size
    base_config["model"]["train_ds"]["kspace_crop"] = kspace_crop
    base_config["model"]["train_ds"]["crop_before_masking"] = crop_before_masking
    base_config["model"]["train_ds"]["kspace_zero_filling_size"] = kspace_zero_filling_size
    base_config["model"]["train_ds"]["normalize_inputs"] = normalize_inputs
    base_config["model"]["train_ds"]["max_norm"] = max_normalization_value
    base_config["model"]["train_ds"]["fft_centered"] = fft_centered
    base_config["model"]["train_ds"]["fft_normalization"] = fft_normalization
    base_config["model"]["train_ds"]["spatial_dims"][0] = first_spatial_dim
    base_config["model"]["train_ds"]["spatial_dims"][1] = second_spatial_dim
    base_config["model"]["train_ds"]["coil_dim"] = coil_dim
    base_config["model"]["train_ds"]["use_seed"] = True
    base_config["model"]["train_ds"]["batch_size"] = batch_size
    base_config["model"]["train_ds"]["shuffle"] = True
    base_config["model"]["train_ds"]["num_workers"] = num_workers
    base_config["model"]["train_ds"]["pin_memory"] = pin_memory
    base_config["model"]["train_ds"]["drop_last"] = drop_last
    if task in ["segmentation", "multitask"]:
        base_config["model"]["train_ds"]["segmentations_path"] = train_segmentations_path
        base_config["model"]["train_ds"]["initial_predictions_path"] = train_initial_predictions_path
        base_config["model"]["train_ds"]["complex_data"] = complex_data
        base_config["model"]["train_ds"]["segmentation_classes"] = segmentation_classes
        base_config["model"]["train_ds"]["segmentation_classes_to_remove"] = segmentation_classes_to_remove
        base_config["model"]["train_ds"]["segmentation_classes_to_combine"] = segmentation_classes_to_combine
        base_config["model"]["train_ds"]["segmentation_classes_to_separate"] = segmentation_classes_to_separate
        base_config["model"]["train_ds"]["segmentation_classes_thresholds"] = segmentation_classes_thresholds
    base_config["model"]["validation_ds"]["data_path"] = val_data_path
    base_config["model"]["validation_ds"]["coil_sensitivity_maps_path"] = val_coil_sensitivity_maps_path
    base_config["model"]["validation_ds"]["mask_path"] = val_mask_path
    base_config["model"]["validation_ds"]["dataset_format"] = dataset_type
    base_config["model"]["validation_ds"]["sample_rate"] = sample_rate
    base_config["model"]["validation_ds"]["volume_sample_rate"] = volume_sample_rate
    base_config["model"]["validation_ds"]["use_dataset_cache"] = use_dataset_cache
    base_config["model"]["validation_ds"]["dataset_cache_file"] = dataset_cache_file
    base_config["model"]["validation_ds"]["num_cols"] = num_cols
    base_config["model"]["validation_ds"]["consecutive_slices"] = consecutive_slices
    base_config["model"]["validation_ds"]["data_saved_per_slice"] = data_saved_per_slice
    base_config["model"]["validation_ds"]["apply_prewhitening"] = apply_prewhitening
    base_config["model"]["validation_ds"]["prewhitening_scale_factor"] = prewhitening_scale_factor
    base_config["model"]["validation_ds"]["prewhitening_patch_start"] = prewhitening_patch_start
    base_config["model"]["validation_ds"]["prewhitening_patch_length"] = prewhitening_patch_length
    base_config["model"]["validation_ds"]["apply_gcc"] = apply_gcc
    base_config["model"]["validation_ds"]["gcc_virtual_coils"] = gcc_virtual_coils
    base_config["model"]["validation_ds"]["gcc_calib_lines"] = gcc_calib_lines
    base_config["model"]["validation_ds"]["gcc_align_data"] = gcc_align_data
    base_config["model"]["validation_ds"]["coil_combination_method"] = coil_combination_method
    base_config["model"]["validation_ds"]["dimensionality"] = dimensionality
    base_config["model"]["validation_ds"]["mask_args"] = {}
    base_config["model"]["validation_ds"]["mask_args"]["type"] = subsampling_type
    if subsampling_type is not None:
        base_config["model"]["validation_ds"]["mask_args"]["accelerations"] = {}
        base_config["model"]["validation_ds"]["mask_args"]["accelerations"] = [x for x in subsampling_accelerations]
        base_config["model"]["validation_ds"]["mask_args"]["center_fractions"] = {}
        base_config["model"]["validation_ds"]["mask_args"]["center_fractions"] = [
            x for x in subsampling_center_fractions
        ]
        base_config["model"]["validation_ds"]["mask_args"]["scale"] = subsampling_center_scale
    base_config["model"]["validation_ds"]["mask_args"]["shift_mask"] = subsampling_shift
    base_config["model"]["validation_ds"]["mask_args"]["use_seed"] = False
    base_config["model"]["validation_ds"]["half_scan_percentage"] = half_scan_percentage
    base_config["model"]["validation_ds"]["remask"] = subsampling_remask
    base_config["model"]["validation_ds"]["crop_size"] = crop_size
    base_config["model"]["validation_ds"]["kspace_crop"] = kspace_crop
    base_config["model"]["validation_ds"]["crop_before_masking"] = crop_before_masking
    base_config["model"]["validation_ds"]["kspace_zero_filling_size"] = kspace_zero_filling_size
    base_config["model"]["validation_ds"]["normalize_inputs"] = normalize_inputs
    base_config["model"]["validation_ds"]["max_norm"] = max_normalization_value
    base_config["model"]["validation_ds"]["fft_centered"] = fft_centered
    base_config["model"]["validation_ds"]["fft_normalization"] = fft_normalization
    base_config["model"]["validation_ds"]["spatial_dims"][0] = first_spatial_dim
    base_config["model"]["validation_ds"]["spatial_dims"][1] = second_spatial_dim
    base_config["model"]["validation_ds"]["coil_dim"] = coil_dim
    base_config["model"]["validation_ds"]["use_seed"] = False
    base_config["model"]["validation_ds"]["batch_size"] = batch_size
    base_config["model"]["validation_ds"]["shuffle"] = False
    base_config["model"]["validation_ds"]["num_workers"] = num_workers
    base_config["model"]["validation_ds"]["pin_memory"] = pin_memory
    base_config["model"]["validation_ds"]["drop_last"] = drop_last
    if task in ["segmentation", "multitask"]:
        base_config["model"]["validation_ds"]["segmentations_path"] = val_segmentations_path
        base_config["model"]["validation_ds"]["initial_predictions_path"] = val_initial_predictions_path
        base_config["model"]["validation_ds"]["complex_data"] = complex_data
        base_config["model"]["validation_ds"]["segmentation_classes"] = segmentation_classes
        base_config["model"]["validation_ds"]["segmentation_classes_to_remove"] = segmentation_classes_to_remove
        base_config["model"]["validation_ds"]["segmentation_classes_to_combine"] = segmentation_classes_to_combine
        base_config["model"]["validation_ds"]["segmentation_classes_to_separate"] = segmentation_classes_to_separate
        base_config["model"]["validation_ds"]["segmentation_classes_thresholds"] = segmentation_classes_thresholds
else:
    base_config["model"]["test_ds"]["data_path"] = test_data_path
    base_config["model"]["test_ds"]["coil_sensitivity_maps_path"] = test_coil_sensitivity_maps_path
    base_config["model"]["test_ds"]["mask_path"] = test_mask_path
    base_config["model"]["test_ds"]["dataset_format"] = dataset_type
    base_config["model"]["test_ds"]["sample_rate"] = sample_rate
    base_config["model"]["test_ds"]["volume_sample_rate"] = volume_sample_rate
    base_config["model"]["test_ds"]["use_dataset_cache"] = use_dataset_cache
    base_config["model"]["test_ds"]["dataset_cache_file"] = dataset_cache_file
    base_config["model"]["test_ds"]["num_cols"] = num_cols
    base_config["model"]["test_ds"]["consecutive_slices"] = consecutive_slices
    base_config["model"]["test_ds"]["data_saved_per_slice"] = data_saved_per_slice
    base_config["model"]["test_ds"]["apply_prewhitening"] = apply_prewhitening
    base_config["model"]["test_ds"]["prewhitening_scale_factor"] = prewhitening_scale_factor
    base_config["model"]["test_ds"]["prewhitening_patch_start"] = prewhitening_patch_start
    base_config["model"]["test_ds"]["prewhitening_patch_length"] = prewhitening_patch_length
    base_config["model"]["test_ds"]["apply_gcc"] = apply_gcc
    base_config["model"]["test_ds"]["gcc_virtual_coils"] = gcc_virtual_coils
    base_config["model"]["test_ds"]["gcc_calib_lines"] = gcc_calib_lines
    base_config["model"]["test_ds"]["gcc_align_data"] = gcc_align_data
    base_config["model"]["test_ds"]["coil_combination_method"] = coil_combination_method
    base_config["model"]["test_ds"]["dimensionality"] = dimensionality
    base_config["model"]["test_ds"]["mask_args"] = {}
    base_config["model"]["test_ds"]["mask_args"]["type"] = subsampling_type
    if subsampling_type is not None:
        base_config["model"]["test_ds"]["mask_args"]["accelerations"] = {}
        base_config["model"]["test_ds"]["mask_args"]["accelerations"] = [x for x in subsampling_accelerations]
        base_config["model"]["test_ds"]["mask_args"]["center_fractions"] = {}
        base_config["model"]["test_ds"]["mask_args"]["center_fractions"] = [x for x in subsampling_center_fractions]
        base_config["model"]["test_ds"]["mask_args"]["scale"] = subsampling_center_scale
    base_config["model"]["test_ds"]["mask_args"]["shift_mask"] = subsampling_shift
    base_config["model"]["test_ds"]["mask_args"]["use_seed"] = False
    base_config["model"]["test_ds"]["half_scan_percentage"] = half_scan_percentage
    base_config["model"]["test_ds"]["remask"] = subsampling_remask
    base_config["model"]["test_ds"]["crop_size"] = crop_size
    base_config["model"]["test_ds"]["kspace_crop"] = kspace_crop
    base_config["model"]["test_ds"]["crop_before_masking"] = crop_before_masking
    base_config["model"]["test_ds"]["kspace_zero_filling_size"] = kspace_zero_filling_size
    base_config["model"]["test_ds"]["normalize_inputs"] = normalize_inputs
    base_config["model"]["test_ds"]["max_norm"] = max_normalization_value
    base_config["model"]["test_ds"]["fft_centered"] = fft_centered
    base_config["model"]["test_ds"]["fft_normalization"] = fft_normalization
    base_config["model"]["test_ds"]["spatial_dims"][0] = first_spatial_dim
    base_config["model"]["test_ds"]["spatial_dims"][1] = second_spatial_dim
    base_config["model"]["test_ds"]["coil_dim"] = coil_dim
    base_config["model"]["test_ds"]["use_seed"] = False
    base_config["model"]["test_ds"]["batch_size"] = batch_size
    base_config["model"]["test_ds"]["shuffle"] = False
    base_config["model"]["test_ds"]["num_workers"] = num_workers
    base_config["model"]["test_ds"]["pin_memory"] = pin_memory
    base_config["model"]["test_ds"]["drop_last"] = drop_last
    if task in ["segmentation", "multitask"]:
        base_config["model"]["test_ds"]["segmentations_path"] = test_segmentations_path
        base_config["model"]["test_ds"]["initial_predictions_path"] = test_initial_predictions_path
        base_config["model"]["test_ds"]["complex_data"] = complex_data
        base_config["model"]["test_ds"]["segmentation_classes"] = segmentation_classes
        base_config["model"]["test_ds"]["segmentation_classes_to_remove"] = segmentation_classes_to_remove
        base_config["model"]["test_ds"]["segmentation_classes_to_combine"] = segmentation_classes_to_combine
        base_config["model"]["test_ds"]["segmentation_classes_to_separate"] = segmentation_classes_to_separate
        base_config["model"]["test_ds"]["segmentation_classes_thresholds"] = segmentation_classes_thresholds

base_config["model"]["optim"]["name"] = optimizer.lower()
base_config["model"]["optim"]["lr"] = learning_rate
base_config["model"]["optim"]["betas"] = betas
base_config["model"]["optim"]["weight_decay"] = weight_decay
base_config["model"]["optim"]["sched"]["name"] = scheduler
base_config["model"]["optim"]["sched"]["min_lr"] = scheduler_min_lr
base_config["model"]["optim"]["sched"]["last_epoch"] = scheduler_last_epoch
base_config["model"]["optim"]["sched"]["warmup_ratio"] = scheduler_warmup_ratio
base_config["trainer"]["strategy"] = trainer_strategy
base_config["trainer"]["accelerator"] = trainer_accelerator
base_config["trainer"]["devices"] = trainer_devices
base_config["trainer"]["num_nodes"] = trainer_num_nodes
base_config["trainer"]["precision"] = trainer_precision
base_config["trainer"]["max_epochs"] = trainer_max_epochs
base_config["trainer"]["enable_checkpointing"] = trainer_enable_checkpointing
base_config["trainer"]["logger"] = trainer_logger
base_config["trainer"]["log_every_n_steps"] = trainer_log_every_n_steps
base_config["trainer"]["check_val_every_n_epoch"] = trainer_check_val_every_n_epoch
base_config["trainer"]["max_steps"] = trainer_max_steps
base_config["exp_manager"]["create_tensorboard_logger"] = create_tensorboard_logger
base_config["exp_manager"]["create_wandb_logger"] = create_wandb_logger
base_config["exp_manager"]["exp_dir"] = export_path
base_config["exp_manager"]["ema"] = {}
base_config["exp_manager"]["ema"]["enable"] = ema

# replace any empty or "None" or "none" or null or "null" values with None
for key, value in base_config.items():
    if value == "" or value == "None" or value == "none" or value == "null" or value == "Null":
        base_config[key] = None

base_config_path = os.path.join(export_path, f"{export_path}/{model_name.lower()}_{mode}.yaml")

if st.button("Export Configuration"):
    if os.path.exists(base_config_path):
        os.remove(base_config_path)
    # log config to yaml
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)
    # show exported path
    st.write(f"Configuration exported to {base_config_path}")

if st.button("Open Tensorboard"):
    # run tensorboard in random port number
    port_idx = 6009
    # open terminal in new window with black background
    os.system(f"gnome-terminal -- tensorboard --logdir {export_path} --port {port_idx}")
    # detach from terminal
    os.system("exit")
    new = 2
    url = f"http://localhost:{port_idx}/"
    # wait for 4 seconds and show message that tensorboard is opening
    st.write("Opening Tensorboard...")
    time.sleep(4)
    webbrowser.open(url, new=new)

# add button next to previous button
if st.button("Run"):
    if os.path.exists(base_config_path):
        os.remove(base_config_path)
    # log config to yaml
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)
    # show exported path
    st.write(f"Configuration exported to {base_config_path}")

    # button to copy command to clipboard
    st.code(f"mridc run -c {export_path}/{model_name.lower()}_{mode}.yaml", language="bash")

    # open terminal in new window with black background
    os.system(f"gnome-terminal -- mridc run -c {export_path}/{model_name.lower()}_{mode}.yaml")
    # detach from terminal
    os.system("exit")
