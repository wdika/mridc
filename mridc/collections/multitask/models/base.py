# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import mridc.collections.common.parts.utils as utils
import mridc.collections.multitask.data.mri_data as multitask_mri_data
import mridc.collections.reconstruction.data.subsample as subsample
import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.collections.multitask.parts.transforms as transforms

__all__ = ["BaseMTLMRIModel"]


class BaseMTLMRIModel(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
    """Base class of all MultiTask Learning MRI models."""

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """
        Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg: Configuration file.
            dict

        Returns
        -------
        dataloader: DataLoader.
            torch.utils.data.DataLoader
        """
        mask_root = cfg.get("mask_path")
        mask_args = cfg.get("mask_args")
        shift_mask = mask_args.get("shift_mask")
        mask_type = mask_args.get("type")

        mask_func = None  # type: ignore
        mask_center_scale = 0.02

        if utils.is_none(mask_root) and not utils.is_none(mask_type):
            accelerations = mask_args.get("accelerations")
            center_fractions = mask_args.get("center_fractions")
            mask_center_scale = mask_args.get("scale")

            mask_func = (
                [
                    subsample.create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) >= 2
                else [subsample.create_mask_for_mask_type(mask_type, center_fractions, accelerations)]
            )

        complex_data = cfg.get("complex_data", True)

        dataset = multitask_mri_data.SKMTEADataset(
            root=cfg.get("data_path"),
            mask_root=mask_root,
            annotations_root=cfg.get("annotations_path"),
            segmentations_root=cfg.get("segmentations_path"),
            initial_predictions_root=cfg.get("initial_predictions_path"),
            split=cfg.get("split"),
            sample_rate=cfg.get("sample_rate", 1.0),
            volume_sample_rate=cfg.get("volume_sample_rate", None),
            use_dataset_cache=cfg.get("use_dataset_cache", None),
            dataset_cache_file=cfg.get("dataset_cache_file", None),
            num_cols=cfg.get("num_cols", None),
            consecutive_slices=cfg.get("consecutive_slices", 1),
            segmentation_classes=cfg.get("segmentation_classes", 2),
            complex_data=complex_data,
            data_saved_per_slice=cfg.get("data_saved_per_slice", False),
            transform=transforms.MTLMRIDataTransforms(
                complex_data=complex_data,
                apply_prewhitening=cfg.get("apply_prewhitening", False),
                prewhitening_scale_factor=cfg.get("prewhitening_scale_factor", 1.0),
                prewhitening_patch_start=cfg.get("prewhitening_patch_start", 10),
                prewhitening_patch_length=cfg.get("prewhitening_patch_length", 30),
                apply_gcc=cfg.get("apply_gcc", False),
                gcc_virtual_coils=cfg.get("gcc_virtual_coils", 10),
                gcc_calib_lines=cfg.get("gcc_calib_lines", 24),
                gcc_align_data=cfg.get("gcc_align_data", True),
                coil_combination_method=cfg.get("coil_combination_method", "SENSE"),
                dimensionality=cfg.get("dimensionality", 2),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                half_scan_percentage=cfg.get("half_scan_percentage", 0.0),
                remask=cfg.get("remask", False),
                crop_size=cfg.get("crop_size", None),
                kspace_crop=cfg.get("kspace_crop", False),
                crop_before_masking=cfg.get("crop_before_masking", True),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size", None),
                normalize_inputs=cfg.get("normalize_inputs", False),
                max_norm=cfg.get("max_norm", True),
                fft_centered=cfg.get("fft_centered", False),
                fft_normalization=cfg.get("fft_normalization", "ortho"),
                spatial_dims=cfg.get("spatial_dims", [-2, -1]),
                coil_dim=cfg.get("coil_dim", 0),
                echo_type=cfg.get("echo_type", "echo1-echo2-mc"),
                consecutive_slices=cfg.get("consecutive_slices", 1),
                use_seed=cfg.get("use_seed", True),
            ),
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size"),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
