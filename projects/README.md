## **Reconstruction**

For training/running a model, you will need to set up a configuration file.
Please check the [model_zoo](projects/reconstruction/model_zoo/conf) for example configurations.
Then, you can run the training/running script with the following command:

`python -m mridc.launch --config-path path_to_config --config-name config.yaml`

For example, to train a CIRIM:

`python -m mridc.launch --config-path mridc/projects/reconstruction/model_zoo/conf/ --config-name base_cirim_train.yaml`

### Datasets

The recommended public datasets to use with this repo for accelerated MRI reconstruction are the:

- [fastMRI](http://arxiv.org/abs/1811.08839) dataset, and the
- [Fully Sampled Knees](http://old.mridata.org/fullysampled/knees/) dataset.

## **Quantitative**

For training/running a model, you will need to set up a configuration file.
Please check the [model_zoo](projects/quantitative/model_zoo/conf) for example configurations.
Then, you can run the training/running script with the following command:

`python -m mridc.launch --config-path path_to_config --config-name config.yaml`

For example, to train a qRIM:

`python -m mridc.launch --config-path mridc/projects/quantitative/model_zoo/conf/ --config-name base_qrim_train.yaml`

### Datasets
The recommended public dataset to use with this repo for quantitative imaging is the [AHEAD](https://doi.org/10.34894/IHZGQM) dataset.
The raw data are in NIfTI format; including 3D complex data, coil sensitivity maps, and brain mask (needed for constraining the loss in training).
The data should be saved as 2D slices and converted in h5 format.
Please run the [preprocessing script](projects/quantitative/datasets/ahead/preprocessing.py) and the [reformat script](projects/quantitative/datasets/ahead/reformat.py) to prepare the data for training and testing.

Note that in the "_A unified model for reconstruction and R2* mapping of accelerated 7T data using the quantitative Recurrent Inference Machine_" paper, the RIM image reconstruction + least squares fitting is performed as initialization of the parameters for qRIM. The qRIM code can process the least squares fitting, however, the reconstructed images are expected to be provided for data loading.

---
## Notes
For running Parallel-Imaging Compressed Sensing (PICS) you need to install the
[BART](https://mrirecon.github.io/bart/). Important! To be able to run BART with a GPU, make sure to compile it with
NVCC.

Unfortunately, the BART package does not support straight-forward import.
So, few things are needed to run BART:

-  After installation, set the TOOLBOX_PATH and PYTHONPATH environment variables for BART on projects/reconstruction/export_bart_path.sh .
-  Finally export tha paths included in the following script (running the script won't work):

    ```
    bash ./projects/export_bart_path.sh
    ```
  Apparently, this need to be done every time you want to run BART.
- Uncomment lines 7 and 114-117 in the [pics module](mridc/collections/reconstruction/models/pics.py).
