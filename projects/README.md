### **Reconstruction**

For training/running a model, you will need to set up a configuration file.
Please check the [model_zoo](projects/reconstruction/model_zoo/conf) for example configurations.
Then, you can run the training/running script with the following command:

`python -m mridc.launch --config-path path_to_config --config-name config.yaml`

For example, to train a CIRIM:

`python -m mridc.launch --config-path mridc/projects/reconstruction/model_zoo/conf/ --config-name base_cirim_train.yaml`

---
For running Parallel-Imaging Compressed Sensing (PICS) you need to install the
[BART](https://mrirecon.github.io/bart/). Important! To be able to run BART with a GPU, make sure to compile it with
NVCC.

Unfortunately, the BART package does not support straight-forward import.
So, few things are needed to run BART:
- After installation, set the TOOLBOX_PATH and PYTHONPATH environment variables for BART on the projects/reconstruction/export_bart_path.sh .
- Finally export tha paths included in the following script (running the script won't work):

    ```
    bash ./projects/reconstruction/export_bart_path.sh
    ```
  Apparently, this need to be done every time you want to run BART.
- Uncomment lines 7 and 114-117 in the [pics module](mridc/collections/reconstruction/models/pics.py).
