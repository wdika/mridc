from pathlib import Path

import h5py
import numpy as np
import pydicom as dicom
from PIL import Image
from tqdm import tqdm

subj01_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj01/series1/flair.dcm"
)
subj01_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj01/series2/flair.dcm"
)

subj02_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj02/series1/flair.dcm"
)
subj02_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj02/series2/flair.dcm"
)

subj03_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj03/series1/flair.dcm"
)
subj03_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj03/series2/flair.dcm"
)

subj04_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj04/series1/flair.dcm"
)
subj04_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj04/series2/flair.dcm"
)

subj05_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj05/series1/flair.dcm"
)
subj05_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj05/series2/flair.dcm"
)

subj06_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj06/series1/flair.dcm"
)
subj06_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj06/series2/flair.dcm"
)

subj07_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj07/series1/flair.dcm"
)
subj07_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj07/series2/flair.dcm"
)

subj08_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj08/series1/flair.dcm"
)
subj08_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj08/series2/flair.dcm"
)

subj09_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj09/series1/flair.dcm"
)
subj09_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj09/series2/flair.dcm"
)

subj10_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj10/series1/flair.dcm"
)
subj10_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj10/series2/flair.dcm"
)

subj11_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj11/series1/flair.dcm"
)
subj11_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj11/series2/flair.dcm"
)

subj12_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj12/series1/flair.dcm"
)
subj12_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj12/series2/flair.dcm"
)

subj13_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj13/series1/flair.dcm"
)
subj13_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj13/series2/flair.dcm"
)

subj14_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj14/series1/flair.dcm"
)
subj14_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj14/series2/flair.dcm"
)

subj15_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj15/series1/flair.dcm"
)
subj15_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj15/series2/flair.dcm"
)

subj16_pics = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj16/series1/flair.dcm"
)
subj16_cirim = (
    "/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj16/series2/flair.dcm"
)

subjs = {
    "subj01": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1009_stroke/4RIM/proc/flair",
        "pics": subj01_pics,
        "cirim": subj01_cirim,
    },
    "subj02": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1139_FlairCS/4RIM/proc/flair",
        "pics": subj02_pics,
        "cirim": subj02_cirim,
    },
    "subj03": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1453_stroke/4RIM/proc/flair",
        "pics": subj03_pics,
        "cirim": subj03_cirim,
    },
    "subj04": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1853_stroke/4RIM/proc/flair",
        "pics": subj04_pics,
        "cirim": subj04_cirim,
    },
    "subj05": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210804_0934_stroke/4RIM/proc/flair",
        "pics": subj05_pics,
        "cirim": subj05_cirim,
    },
    "subj06": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210806_0858_flairCS/4RIM/proc/flair",
        "pics": subj06_pics,
        "cirim": subj06_cirim,
    },
    "subj07": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210806_1539_stroke/4RIM/proc/flair",
        "pics": subj07_pics,
        "cirim": subj07_cirim,
    },
    "subj08": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210810_1746_stroke/4RIM/proc/flair",
        "pics": subj08_pics,
        "cirim": subj08_cirim,
    },
    "subj09": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220202_dn_fmri/4RIM/proc/flair",
        "pics": subj09_pics,
        "cirim": subj09_cirim,
    },
    "subj10": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220202_ee_covid/4RIM/proc/flair",
        "pics": subj10_pics,
        "cirim": subj10_cirim,
    },
    "subj11": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220211_as_SEP/4RIM/proc/flair",
        "pics": subj11_pics,
        "cirim": subj11_cirim,
    },
    "subj12": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220214_pv_SEP/4RIM/proc/flair",
        "pics": subj12_pics,
        "cirim": subj12_cirim,
    },
    "subj13": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220222_ae-sep/4RIM/proc/flair",
        "pics": subj13_pics,
        "cirim": subj13_cirim,
    },
    "subj14": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220222_dl_sep/4RIM/proc/flair",
        "pics": subj14_pics,
        "cirim": subj14_cirim,
    },
    "subj15": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220223_dy_sep/4RIM/proc/flair",
        "pics": subj15_pics,
        "cirim": subj15_cirim,
    },
    "subj16": {
        "data": "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220223_tf_sep/4RIM/proc/flair",
        "pics": subj16_pics,
        "cirim": subj16_cirim,
    },
}

out_dir = "/data/projects/recon/other/dkarkalousos/STAIRS/png/"

for subj_name, subj_info in tqdm(subjs.items()):
    if subj_name in ("subj09", "subj10", "subj11", "subj12", "subj13", "subj14", "subj15", "subj16"):
        data = h5py.File(subj_info["data"], "r")
        kspace = data["kspace"][()]
        sensitivity_map = data["sensitivity_map"][()]
        axial_imspace = np.fft.ifftn(kspace, axes=(-2, -1))
        axial_target = np.abs(np.sum(axial_imspace * sensitivity_map.conj(), 1))
        axial_target = axial_target / np.max(axial_target)
        axial_target = np.flip(axial_target, axis=1)

        full_kspace = np.fft.fft(kspace, axis=0)
        coronal_imspace = np.fft.ifftn(np.transpose(full_kspace, (2, 1, 0, 3)), axes=(0, 2, 3))
        coronal_sensitivity_map = np.transpose(sensitivity_map, (2, 1, 0, 3))
        coronal_target = np.abs(np.sum(coronal_imspace * coronal_sensitivity_map.conj(), 1))
        coronal_target = coronal_target / np.max(coronal_target)
        coronal_target = np.flip(coronal_target, axis=0)

        sagittal_imspace = np.fft.ifftn(np.transpose(full_kspace, (3, 1, 0, 2)), axes=(0, 2, 3))
        sagittal_sensitivity_map = np.transpose(sensitivity_map, (3, 1, 0, 2))
        sagittal_target = np.abs(np.sum(sagittal_imspace * sagittal_sensitivity_map.conj(), 1))
        sagittal_target = sagittal_target / np.max(sagittal_target)
        sagittal_target = np.flip(sagittal_target, axis=2)

        axial_dir = Path(out_dir + "/" + subj_name + "/axial/")
        axial_dir.mkdir(parents=True, exist_ok=True)

        axial_pics = np.array(dicom.dcmread(subj_info["pics"]).pixel_array)
        axial_pics = axial_pics / np.max(axial_pics)
        init_axial_pics = axial_pics.copy()

        axial_cirim = np.array(dicom.dcmread(subj_info["cirim"]).pixel_array)
        axial_cirim = axial_cirim / np.max(axial_cirim)
        init_axial_cirim = axial_cirim.copy()

        axial_pics = np.flip(axial_pics, axis=0)
        axial_pics = np.flip(axial_pics, axis=1)
        axial_cirim = np.flip(axial_cirim, axis=0)
        axial_cirim = np.flip(axial_cirim, axis=1)

        for i in tqdm(range(axial_pics.shape[0])):
            img = np.concatenate((axial_target[i], axial_pics[i], axial_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(axial_dir) + "/" + str(i) + ".png")

        coronal_dir = Path(out_dir + "/" + subj_name + "/coronal/")
        coronal_dir.mkdir(parents=True, exist_ok=True)

        coronal_pics = np.transpose(init_axial_pics, (1, 0, 2))  # coronal
        coronal_cirim = np.transpose(init_axial_cirim, (1, 0, 2))  # coronal

        coronal_target = np.flip(coronal_target, axis=0)
        coronal_pics = np.flip(coronal_pics, axis=1)
        coronal_cirim = np.flip(coronal_cirim, axis=1)

        for i in tqdm(range(coronal_pics.shape[0])):
            img = np.concatenate((coronal_target[i], coronal_pics[i], coronal_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(coronal_dir) + "/" + str(i) + ".png")

        sagittal_dir = Path(out_dir + "/" + subj_name + "/sagittal/")
        sagittal_dir.mkdir(parents=True, exist_ok=True)

        sagittal_pics = np.transpose(init_axial_pics, (2, 0, 1))  # sagittal
        sagittal_cirim = np.transpose(init_axial_cirim, (2, 0, 1))  # sagittal

        sagittal_pics = np.flip(sagittal_pics, axis=0)
        sagittal_pics = np.flip(sagittal_pics, axis=1)
        sagittal_pics = np.flipud(sagittal_pics)
        sagittal_pics = np.flip(sagittal_pics, axis=2)
        sagittal_cirim = np.flip(sagittal_cirim, axis=0)
        sagittal_cirim = np.flip(sagittal_cirim, axis=1)
        sagittal_cirim = np.flipud(sagittal_cirim)
        sagittal_cirim = np.flip(sagittal_cirim, axis=2)

        for i in tqdm(range(sagittal_pics.shape[0])):
            img = np.concatenate((sagittal_target[i], sagittal_pics[i], sagittal_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(sagittal_dir) + "/" + str(i) + ".png")
    elif subj_name == "subj06":
        data = h5py.File(subj_info["data"], "r")
        kspace = data["kspace"][()]
        sensitivity_map = data["sensitivity_map"][()]
        coronal_imspace = np.fft.ifftn(kspace, axes=(-2, -1))
        coronal_target = np.abs(np.sum(coronal_imspace * sensitivity_map.conj(), 1))
        coronal_target = coronal_target / np.max(coronal_target)
        coronal_target = np.flip(coronal_target, axis=0)
        coronal_target = np.flip(coronal_target, axis=1)

        full_kspace = np.fft.fft(kspace, axis=0)
        sagittal_imspace = np.fft.ifftn(np.transpose(full_kspace, (2, 1, 0, 3)), axes=(0, 2, 3))
        sagittal_sensitivity_map = np.transpose(sensitivity_map, (2, 1, 0, 3))
        sagittal_target = np.abs(np.sum(sagittal_imspace * sagittal_sensitivity_map.conj(), 1))
        sagittal_target = sagittal_target / np.max(sagittal_target)
        sagittal_target = np.flip(sagittal_target, axis=0)
        sagittal_target = np.flip(sagittal_target, axis=1)

        axial_imspace = np.fft.ifftn(np.transpose(full_kspace, (3, 1, 0, 2)), axes=(0, 2, 3))
        axial_sensitivity_map = np.transpose(sensitivity_map, (3, 1, 0, 2))
        axial_target = np.abs(np.sum(axial_imspace * axial_sensitivity_map.conj(), 1))
        axial_target = axial_target / np.max(axial_target)
        axial_target = np.flip(axial_target, axis=2)

        coronal_dir = Path(out_dir + "/" + subj_name + "/coronal/")
        coronal_dir.mkdir(parents=True, exist_ok=True)

        coronal_pics = np.array(dicom.dcmread(subj_info["pics"]).pixel_array)
        coronal_pics = coronal_pics / np.max(coronal_pics)

        coronal_cirim = np.array(dicom.dcmread(subj_info["cirim"]).pixel_array)
        coronal_cirim = coronal_cirim / np.max(coronal_cirim)

        for i in tqdm(range(coronal_pics.shape[0])):
            img = np.concatenate((coronal_target[i], coronal_pics[i], coronal_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(coronal_dir) + "/" + str(i) + ".png")

        sagittal_dir = Path(out_dir + "/" + subj_name + "/sagittal/")
        sagittal_dir.mkdir(parents=True, exist_ok=True)

        sagittal_pics = np.transpose(coronal_pics, (1, 0, 2))  # sagittal
        sagittal_cirim = np.transpose(coronal_cirim, (1, 0, 2))  # sagittal

        for i in tqdm(range(sagittal_pics.shape[0])):
            img = np.concatenate((sagittal_target[i], sagittal_pics[i], sagittal_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(sagittal_dir) + "/" + str(i) + ".png")

        axial_dir = Path(out_dir + "/" + subj_name + "/axial/")
        axial_dir.mkdir(parents=True, exist_ok=True)

        axial_pics = np.transpose(coronal_pics, (2, 0, 1))  # axial
        axial_pics = np.flip(axial_pics, axis=1)
        axial_cirim = np.transpose(coronal_cirim, (2, 0, 1))  # axial
        axial_cirim = np.flip(axial_cirim, axis=1)

        for i in tqdm(range(axial_pics.shape[0])):
            img = np.concatenate((axial_target[i], axial_pics[i], axial_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(axial_dir) + "/" + str(i) + ".png")
    else:
        data = h5py.File(subj_info["data"], "r")
        kspace = data["kspace"][()]
        sensitivity_map = data["sensitivity_map"][()]
        axial_imspace = np.fft.ifftn(kspace, axes=(-2, -1))
        axial_target = np.abs(np.sum(axial_imspace * sensitivity_map.conj(), 1))
        axial_target = axial_target / np.max(axial_target)
        axial_target = np.flip(axial_target, axis=1)

        full_kspace = np.fft.fft(kspace, axis=0)
        coronal_imspace = np.fft.ifftn(np.transpose(full_kspace, (2, 1, 0, 3)), axes=(0, 2, 3))
        coronal_sensitivity_map = np.transpose(sensitivity_map, (2, 1, 0, 3))
        coronal_target = np.abs(np.sum(coronal_imspace * coronal_sensitivity_map.conj(), 1))
        coronal_target = coronal_target / np.max(coronal_target)
        coronal_target = np.flip(coronal_target, axis=0)

        sagittal_imspace = np.fft.ifftn(np.transpose(full_kspace, (3, 1, 0, 2)), axes=(0, 2, 3))
        sagittal_sensitivity_map = np.transpose(sensitivity_map, (3, 1, 0, 2))
        sagittal_target = np.abs(np.sum(sagittal_imspace * sagittal_sensitivity_map.conj(), 1))
        sagittal_target = sagittal_target / np.max(sagittal_target)
        sagittal_target = np.flip(sagittal_target, axis=2)

        axial_dir = Path(out_dir + "/" + subj_name + "/axial/")
        axial_dir.mkdir(parents=True, exist_ok=True)

        axial_pics = np.array(dicom.dcmread(subj_info["pics"]).pixel_array)
        axial_pics = axial_pics / np.max(axial_pics)

        axial_cirim = np.array(dicom.dcmread(subj_info["cirim"]).pixel_array)
        axial_cirim = axial_cirim / np.max(axial_cirim)

        for i in tqdm(range(axial_pics.shape[0])):
            img = np.concatenate((axial_target[i], axial_pics[i], axial_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(axial_dir) + "/" + str(i) + ".png")

        coronal_dir = Path(out_dir + "/" + subj_name + "/coronal/")
        coronal_dir.mkdir(parents=True, exist_ok=True)

        coronal_pics = np.transpose(axial_pics, (1, 0, 2))  # coronal
        coronal_cirim = np.transpose(axial_cirim, (1, 0, 2))  # coronal

        for i in tqdm(range(coronal_pics.shape[0])):
            img = np.concatenate((coronal_target[i], coronal_pics[i], coronal_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(coronal_dir) + "/" + str(i) + ".png")

        sagittal_dir = Path(out_dir + "/" + subj_name + "/sagittal/")
        sagittal_dir.mkdir(parents=True, exist_ok=True)

        sagittal_pics = np.transpose(axial_pics, (2, 0, 1))  # sagittal
        sagittal_cirim = np.transpose(axial_cirim, (2, 0, 1))  # sagittal

        for i in tqdm(range(sagittal_pics.shape[0])):
            img = np.concatenate((sagittal_target[i], sagittal_pics[i], sagittal_cirim[i]), axis=1)
            img = img / np.max(img)
            img = Image.fromarray(np.uint8(img * 255))
            img.save(str(sagittal_dir) + "/" + str(i) + ".png")
