# Data Consistency for Magnetic Resonance Imaging

[![CodeQL](https://github.com/wdika/mridc/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/wdika/mridc/actions/workflows/codeql-analysis.yml)
[![CircleCI](https://circleci.com/gh/wdika/mridc/tree/main.svg?style=svg)](https://circleci.com/gh/wdika/mridc/tree/main)
[![codecov](https://codecov.io/gh/wdika/mridc/branch/main/graph/badge.svg?token=KPPQ33DOTF)](https://codecov.io/gh/wdika/mridc)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

---
## Introduction

**MRIDC is a toolbox for applying AI methods on MR imaging. A collection of tools for data consistency and data quality
is provided for MRI data analysis. Primarily it focuses on the following tasks:**

### **Reconstruction**:
1.[Cascades of Independently Recurrent Inference Machines (CIRIM)](https://iopscience.iop.org/article/10.1088/1361-6560/ac6cc2),
2.[Compressed Sensing (CS)](https://ieeexplore.ieee.org/document/4472246),
3.[Convolutional Recurrent Neural Networks (CRNN)](https://ieeexplore.ieee.org/document/8425639),
4.[Deep Cascade of Convolutional Neural Networks (CCNN)](https://ieeexplore.ieee.org/document/8067520),
5.[Down-Up Net (DUNET)](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28827),
6.[End-to-End Variational Network (E2EVN)](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_7),
7.[Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet)](https://ieeexplore.ieee.org/document/9578412),
8.[Independently Recurrent Inference Machines (IRIM)](http://arxiv.org/abs/2012.07819),
9.[KIKI-Net](https://onlinelibrary.wiley.com/doi/10.1002/mrm.27201),
10.[Learned Primal-Dual Net (LPDNet)](https://ieeexplore.ieee.org/document/8271999),
11.[MultiDomainNet](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8428775/),
12.[Recurrent Inference Machines (RIM)](https://www.sciencedirect.com/science/article/abs/pii/S1361841518306078?via%3Dihub),
13.[Recurrent Variational Network (RVN)](https://arxiv.org/abs/2111.09639),
14.[UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28),
15.[Variable Splitting Network (VSNet)](https://dl.acm.org/doi/abs/10.1007/978-3-030-32251-9_78),
16.[XPDNet](https://arxiv.org/abs/2010.07290),
17.and Zero-Filled reconstruction (ZF).

### **Segmentation**:
_Coming soon..._

### **Acknowledgements**

MRIDC is based on the [NeMo](https://github.com/NVIDIA/NeMo) framework, using PyTorch Lightning for feasible
high-performance multi-GPU/multi-node mixed-precision training.

For the reconstruction methods:
- the implementations of 6 and 14 are thanks to and based on the [fastMRI repo](https://github.com/facebookresearch/fastMRI).
- The implementations of 7, 9, 10, 11, 13, and 16 are thanks to and based on the [DIRECT repo](https://github.com/NKI-AI/direct).

## Installation

MRIDC is best to be installed in a Conda environment.

    conda create -n mridc python=3.9
    conda activate mridc

### Pip

Use pip installation if you want the latest stable version.
```bash
pip install mridc
```

### From source

Use source installation if you want the latest development version, as well as for contributing to MRIDC.

```bash
git clone https://github.com/wdika/mridc
cd mridc
./reinstall.sh
```

## Usage

Check the [projects](https://github.com/wdika/mridc/blob/main/projects/README.md) page for more information of how to use **mridc**.

### Datasets

Recommended public datasets to use with this repo:

- [fastMRI](http://arxiv.org/abs/1811.08839),
- [Fully Sampled Knees](http://old.mridata.org/fullysampled/knees/).

## API Documentation

[![Documentation Status](https://readthedocs.org/projects/mridc/badge/?version=latest)](https://mridc.readthedocs.io/en/latest/?badge=latest)

Access the API Documentation [here](https://mridc.readthedocs.io/en/latest/modules.html)

## License

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Citation

Please cite MRIDC using the "_Cite this repository_" button or as

```BibTeX
@misc{mridc,
    author = {Karkalousos, Dimitrios and Caan, Matthan},
    title = {MRIDC: Data Consistency for Magnetic Resonance Imaging},
    year = {2021},
    url = {https://github.com/wdika/mridc},
}
```

## Papers

The following papers use the MRIDC repo:

[1] [Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent
Inference Machines for fast and robust accelerated MRI reconstruction’](https://iopscience.iop.org/article/10.1088/1361-6560/ac6cc2)
