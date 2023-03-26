# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import pathlib

import torch
from omegaconf import OmegaConf, open_dict


def main():
    trained_model_path = pathlib.Path(args.trained_model_path)

    new_keys = ["val_loss_fn"]

    old_keys = ["eval_loss_fn"]

    for old_key_to_replace, new_key_to_replace in zip(old_keys, new_keys):
        print(f"Replacing {old_key_to_replace} with {new_key_to_replace}")

        mridc_log_file = trained_model_path / "mridc_log_globalrank-0_localrank-0.txt"
        mridc_hparams_file = trained_model_path / "hparams.yaml"

        # Edit mridc_log_globalrank-0_localrank-0.txt
        with open(mridc_log_file, "r") as f:
            lines = f.readlines()
        with open(mridc_log_file, "w") as f:
            for line in lines:
                if old_key_to_replace in line:
                    line = line.replace(old_key_to_replace, new_key_to_replace)
                f.write(line)

        # Edit hparams.yaml
        with open(mridc_hparams_file, "r") as f:
            lines = f.readlines()
        with open(mridc_hparams_file, "w") as f:
            for line in lines:
                if old_key_to_replace in line:
                    line = line.replace(old_key_to_replace, new_key_to_replace)
                f.write(line)

        # checkpoints = list(pathlib.Path(trained_model_path / "checkpoints").iterdir())
        # go on dir back
        checkpoints = list(pathlib.Path(trained_model_path.parent).iterdir())
        for checkpoint in checkpoints:
            if checkpoint.suffix == ".ckpt":
                chckpnt = torch.load(checkpoint, map_location=torch.device("cpu"))

                # replace in hyper_parameters
                cfg = chckpnt["hyper_parameters"]["cfg"]
                new_cfg = OmegaConf.create()
                for key, value in cfg.items():
                    if key == old_key_to_replace:
                        key = new_key_to_replace
                    new_cfg[key] = value
                chckpnt["hyper_parameters"]["cfg"] = new_cfg

                # replace in state_dict
                state_dict = chckpnt["state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if old_key_to_replace in key:
                        key = key.replace(old_key_to_replace, new_key_to_replace)
                    new_state_dict[key] = value
                chckpnt["state_dict"] = new_state_dict

                torch.save(chckpnt, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            python projects/multi_task_learning/scripts/edit_trainedmodel_files.py
            /data/projects/recon/other/dkarkalousos/trained_models/fastMRI_FLAIR_equispaced1d_4x_320x320/CIRIM_5C_64F_ssim_NODC/2022-02-25_13-03-12/
            eval_loss_fn
            val_loss_fn
            """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("trained_model_path", type=pathlib.Path, default=None, help="Path to the trained model.")
    # parser.add_argument("old_key_to_replace", type=str, default=None, help="Old value")
    # parser.add_argument("new_key_to_replace", type=str, default=None, help="New value")
    args = parser.parse_args()
    main()
