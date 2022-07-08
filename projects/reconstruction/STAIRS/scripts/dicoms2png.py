import os
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


subj01_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj01/series1/flair.dcm'
subj01_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj01/series2/flair.dcm'

subj02_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj02/series1/flair.dcm'
subj02_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj02/series2/flair.dcm'

subj03_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj03/series1/flair.dcm'
subj03_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj03/series2/flair.dcm'

subj04_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj04/series1/flair.dcm'
subj04_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj04/series2/flair.dcm'

subj05_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj05/series1/flair.dcm'
subj05_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj05/series2/flair.dcm'

subj06_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj06/series1/flair.dcm'
subj06_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj06/series2/flair.dcm'

subj07_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj07/series1/flair.dcm'
subj07_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj07/series2/flair.dcm'

subj08_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj08/series1/flair.dcm'
subj08_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj08/series2/flair.dcm'

subj09_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj09/series1/flair.dcm'
subj09_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj09/series2/flair.dcm'

subj10_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj10/series1/flair.dcm'
subj10_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj10/series2/flair.dcm'

subj11_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj11/series1/flair.dcm'
subj11_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj11/series2/flair.dcm'

subj12_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj12/series1/flair.dcm'
subj12_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj12/series2/flair.dcm'

subj13_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj13/series1/flair.dcm'
subj13_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj13/series2/flair.dcm'

subj14_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj14/series1/flair.dcm'
subj14_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj14/series2/flair.dcm'

subj15_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj15/series1/flair.dcm'
subj15_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj15/series2/flair.dcm'

subj16_pics = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj16/series1/flair.dcm'
subj16_cirim = '/data/projects/recon/data/private/STAIRS/proc/Rothschild/dicoms_anon_nobiascorrected/subj16/series2/flair.dcm'

subjs = {
    'subj01': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1009_stroke/4RIM/proc/flair",
        'pics': subj01_pics,
        'cirim': subj01_cirim
    },
    'subj02': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1139_FlairCS/4RIM/proc/flair",
        'pics': subj02_pics,
        'cirim': subj02_cirim
    },
    'subj03': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1453_stroke/4RIM/proc/flair",
        'pics': subj03_pics,
        'cirim': subj03_cirim
    },
    'subj04': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210803_1853_stroke/4RIM/proc/flair",
        'pics': subj04_pics,
        'cirim': subj04_cirim
    },
    'subj05': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210804_0934_stroke/4RIM/proc/flair",
        'pics': subj05_pics,
        'cirim': subj05_cirim
    },
    'subj06': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210806_0858_flairCS/4RIM/proc/flair",
        'pics': subj06_pics,
        'cirim': subj06_cirim
    },
    'subj07': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210806_1539_stroke/4RIM/proc/flair",
        'pics': subj07_pics,
        'cirim': subj07_cirim
    },
    'subj08': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20210810_1746_stroke/4RIM/proc/flair",
        'pics': subj08_pics,
        'cirim': subj08_cirim
    },
    'subj09': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220202_dn_fmri/4RIM/proc/flair",
        'pics': subj09_pics,
        'cirim': subj09_cirim
    },
    'subj10': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220202_ee_covid/4RIM/proc/flair",
        'pics': subj10_pics,
        'cirim': subj10_cirim
    },
    'subj11': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220211_as_SEP/4RIM/proc/flair",
        'pics': subj11_pics,
        'cirim': subj11_cirim
    },
    'subj12': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220214_pv_SEP/4RIM/proc/flair",
        'pics': subj12_pics,
        'cirim': subj12_cirim
    },
    'subj13': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220222_ae-sep/4RIM/proc/flair",
        'pics': subj13_pics,
        'cirim': subj13_cirim
    },
    'subj14': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220222_dl_sep/4RIM/proc/flair",
        'pics': subj14_pics,
        'cirim': subj14_cirim
    },
    'subj15': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220223_dy_sep/4RIM/proc/flair",
        'pics': subj15_pics,
        'cirim': subj15_cirim
    },
    'subj16': {
        'data': "/data/projects/recon/data/private/STAIRS/proc/Rothschild/20220223_tf_sep/4RIM/proc/flair",
        'pics': subj16_pics,
        'cirim': subj16_cirim
    }
}

out_dir = "/data/projects/recon/other/dkarkalousos/STAIRS/png/"

for subj_name, subj_info in tqdm(subjs.items()):
    axial_dir = Path(out_dir + "/" + subj_name + "/axial/")
    axial_dir.mkdir(parents=True, exist_ok=True)

    axial_pics = np.array(dicom.dcmread(subj_info['pics']).pixel_array)
    axial_pics = axial_pics / np.max(axial_pics)

    axial_cirim = np.array(dicom.dcmread(subj_info['cirim']).pixel_array)
    axial_cirim = axial_cirim / np.max(axial_cirim)

    # data = h5py.File(subj_info['data'], 'r')
    # kspace = data['kspace'][()]
    # sensitivity_map = data['sensitivity_map'][()]
    # imspace = np.fft.ifftn(kspace, axes=(-2, -1))
    # target = np.abs(np.sum(imspace*sensitivity_map.conj(), 1))
    # target = target / np.max(target)
    # target = np.flip(target, axis=(1))

    for i in tqdm(range(axial_pics.shape[0])):
        # plt.subplot(1, 3, 1)
        # plt.imshow(target[i], cmap='gray')
        # plt.title("Zero-Filled")
        plt.subplot(1, 2, 1)
        plt.imshow(axial_pics[i], cmap='gray')
        plt.title("PICS")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(axial_cirim[i], cmap='gray')
        plt.title("CIRIM")
        plt.axis('off')
        plt.savefig(str(axial_dir) + "/" + str(i) + ".png")

    coronal_dir = Path(out_dir + "/" + subj_name + "/coronal/")
    coronal_dir.mkdir(parents=True, exist_ok=True)

    coronal_pics = np.transpose(axial_pics, (1, 0, 2))  # coronal
    coronal_cirim = np.transpose(axial_cirim, (1, 0, 2))  # coronal

    for i in tqdm(range(coronal_pics.shape[0])):
        plt.subplot(1, 2, 1)
        plt.imshow(coronal_pics[i], cmap='gray')
        plt.title("PICS")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(coronal_cirim[i], cmap='gray')
        plt.title("CIRIM")
        plt.axis('off')
        plt.savefig(str(coronal_dir) + "/" + str(i) + ".png")

    sagittal_dir = Path(out_dir + "/" + subj_name + "/sagittal/")
    sagittal_dir.mkdir(parents=True, exist_ok=True)

    sagittal_pics = np.transpose(axial_pics, (2, 0, 1))  # sagittal
    sagittal_cirim = np.transpose(axial_cirim, (2, 0, 1))  # sagittal

    for i in tqdm(range(sagittal_pics.shape[0])):
        plt.subplot(1, 2, 1)
        plt.imshow(sagittal_pics[i], cmap='gray')
        plt.title("PICS")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(sagittal_cirim[i], cmap='gray')
        plt.title("CIRIM")
        plt.axis('off')
        plt.savefig(str(sagittal_dir) + "/" + str(i) + ".png")
