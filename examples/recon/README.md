For training/running a model, please use the following command:

`python -m examples.recon.launch --config-name config.yaml`

For running Parallel-Imaging Compressed Sensing (PICS) you need to install the
[BART](https://mrirecon.github.io/bart/). Important! To be able to run BART with a GPU, make sure to compile it with
NVCC.

After installation, set the TOOLBOX_PATH and PYTHONPATH environment variables for BART:

```
export TOOLBOX_PATH=/path/to/bart
export PYTHONPATH=${TOOLBOX_PATH}/python/
```
