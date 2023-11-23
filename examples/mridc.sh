# activate conda environment
source activate /scratch/dkarkalousos/envs/mridc/
# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so to avoid libstdc++ error
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so
# cd to the directory where the script is located
cd /home/dkarkalousos/PycharmProjects/mridc/tools
# run the script with streamlit
streamlit run app.py
