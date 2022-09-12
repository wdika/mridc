# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import glob
import sys
from pathlib import Path

import h5py
from tqdm import tqdm


def iterate_qmap(qmap, name):
    qmap_recons = []
    recon_accs = []
    for key, val in qmap:
        if "recon" in key:
            acc = key.split("_")[4]
            recon_accs.append(acc)
            qmap_recons.append([f"{name}_recon_{acc}x", val])

    qmap_inits = []
    qmap_targets = []
    accs = []
    target_saved = False
    for key, val in qmap:
        if "init" in key:
            if len(key.split("_")) > 3:
                acc = key.split("_")[-1]
                if acc not in accs and acc in recon_accs:
                    accs.append(acc)
                    qmap_inits.append([f"{name}_init_{acc}x", val])
        elif "target" in key and not target_saved:
            target_saved = True
            qmap_targets.append([f"{name}_target", val])

    return qmap_recons, qmap_inits, qmap_targets, accs


def main(args):
    out_dir_data = Path(str(args.out_path) + "/multicoil_" + str(args.set) + "/")
    out_dir_data.mkdir(parents=True, exist_ok=True)

    files = list(Path(args.file_path).iterdir())
    files = [files[i] for i in range(len(files)) if "Subcortex" in str(files[i])]
    files = [glob.glob(str(plane) + "/" + "*.h5") for i in range(len(files)) for plane in list(files[i].iterdir())]

    maps = []
    data = []
    kspace_masks = []
    cs = []
    for _file in files:
        for _file_ in _file:
            if "cs" in _file_:
                cs.append(Path(_file_))
            elif "kspmask" in _file_:
                kspace_masks.append(Path(_file_))
            elif "maps" in _file_:
                maps.append(Path(_file_))
            else:
                data.append(Path(_file_))

    maps = sorted(maps)
    data = sorted(data)
    kspace_masks = sorted(kspace_masks)

    data_dict = {}
    if args.set == "test":
        for _data in data:
            fname = str(_data).split("/")[-1].split(".")[0]

            data_dict[fname] = {
                "subject_idx": int(fname.split("_")[-3]),
                "plane": fname.split("_")[-2],
                "slice_idx": int(fname.split("_")[-1]),
                "data": _data,
            }
    else:
        for _maps, _data in zip(maps, data):
            fname = str(_data).split("/")[-1].split(".")[0]

            for _kspmask in kspace_masks:
                kspace_mask = _kspmask if fname in str(_kspmask) else None
            data_dict[fname] = {
                "subject_idx": int(fname.split("_")[-3]),
                "plane": fname.split("_")[-2],
                "slice_idx": int(fname.split("_")[-1]),
                "maps": _maps,
                "data": _data,
                "kspace_mask": kspace_mask,
            }

    for fname in tqdm(data_dict):
        B0_maps = []
        R2star_maps = []
        S0_maps = []
        phi_maps = []

        kspace = None
        mask_brain = None
        mask_head = None
        sense = None

        masks = []
        seeds = []

        if args.set != "test":
            maps = h5py.File(data_dict[fname]["maps"], "r")
            for key in maps.keys():
                if "B0_map" in key:
                    B0_maps.append([key, maps[key][()]])
                elif "R2star_map" in key:
                    R2star_maps.append([key, maps[key][()]])
                elif "S0_map" in key:
                    S0_maps.append([key, maps[key][()]])
                elif "phi_map" in key:
                    phi_maps.append([key, maps[key][()]])
                elif "ksp" in key and kspace is None:
                    kspace = maps[key][()]
                elif "mask_brain" in key and mask_brain is None:
                    mask_brain = maps[key][()]
                elif "mask_head" in key and mask_head is None:
                    mask_head = maps[key][()]
                elif "subsampling_mask" in key:
                    masks.append([key, maps[key][()]])
                elif "seed" in key:
                    seeds.append([key, maps[key][()]])

        data = h5py.File(data_dict[fname]["data"], "r")
        for key in data.keys():
            if "sense" in key and sense is None:  # and idx_plane not in sense_maps_idxs:
                sense = data[key][()]
            elif "B0_map" in key:
                B0_maps.append([key, data[key][()]])
            elif "R2star_map" in key:
                R2star_maps.append([key, data[key][()]])
            elif "S0_map" in key:
                S0_maps.append([key, data[key][()]])
            elif "phi_map" in key:
                phi_maps.append([key, data[key][()]])
            elif "ksp" in key and kspace is None:
                kspace = data[key][()]
            elif "mask_brain" in key and mask_brain is None:
                mask_brain = data[key][()]
            elif "mask_head" in key and mask_head is None:
                mask_head = data[key][()]
            elif "subsampling_mask" in key:
                masks.append([key, data[key][()]])

        R2_star_recons, R2_star_inits, R2_star_targets, R2_star_accs = iterate_qmap(R2star_maps, "R2star_map")

        masks_accs = []
        _masks = []
        for key, val in masks:
            acc = key.split("_")[3]
            if acc not in masks_accs and acc in R2_star_accs:
                masks_accs.append(acc)
                _masks.append([f"mask_{acc}x", val])

        B0_recons, B0_inits, B0_targets, B0_accs = iterate_qmap(B0_maps, "B0_map")
        S0_recons, S0_inits, S0_targets, S0_accs = iterate_qmap(S0_maps, "S0_map")
        phi_recons, phi_inits, phi_targets, phi_accs = iterate_qmap(phi_maps, "phi_map")
        if len(R2_star_recons) != 0 and len(B0_recons) != 0 and len(S0_recons) != 0 and len(phi_recons) != 0:
            hf = h5py.File(Path(f"{str(out_dir_data)}/{fname}"), "w")

            for key, val in R2_star_recons:
                hf.create_dataset(key, data=val)
            for key, val in R2_star_inits:
                hf.create_dataset(key, data=val)
            for key, val in R2_star_targets:
                hf.create_dataset(key, data=val)
            for key, val in B0_recons:
                hf.create_dataset(key, data=val)
            for key, val in B0_inits:
                hf.create_dataset(key, data=val)
            for key, val in B0_targets:
                hf.create_dataset(key, data=val)
            for key, val in S0_recons:
                hf.create_dataset(key, data=val)
            for key, val in S0_inits:
                hf.create_dataset(key, data=val)
            for key, val in S0_targets:
                hf.create_dataset(key, data=val)
            for key, val in phi_recons:
                hf.create_dataset(key, data=val)
            for key, val in phi_inits:
                hf.create_dataset(key, data=val)
            for key, val in phi_targets:
                hf.create_dataset(key, data=val)
            hf.create_dataset("ksp", data=kspace)
            hf.create_dataset("sense", data=sense)
            hf.create_dataset("mask_brain", data=mask_brain)
            if mask_head is not None:
                hf.create_dataset("mask_head", data=mask_head)
            for key, val in _masks:
                hf.create_dataset(key, data=val)
            hf.close()


# noinspection PyTypeChecker
def create_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_path", type=str, help="Path of the files to be converted.")
    parser.add_argument("out_path", type=str, help="Path to save the converted files.")
    parser.add_argument("set", type=str, help="train/val/test")
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
