For training/running a model, please use the following command:

`python -m mridc.launch --config-path path_to_config --config-name config.yaml`

For example, to train a CIRIM:

`python -m mridc.launch --config-path /home/dkarkalousos/PycharmProjects/mridc/projects/reconstruction/model_zoo/conf --config-name base_cirim_train.yaml`


For running Parallel-Imaging Compressed Sensing (PICS) you need to install the
[BART](https://mrirecon.github.io/bart/). Important! To be able to run BART with a GPU, make sure to compile it with
NVCC.

After installation, set the TOOLBOX_PATH and PYTHONPATH environment variables for BART on the projects/reconstruction/export_bart_path.sh .
Finally run the script:

```
bash ./projects/reconstruction/export_bart_path.sh
```
