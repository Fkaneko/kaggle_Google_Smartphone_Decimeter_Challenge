conda install  numpy pandas matplotlib jupyter spyder seaborn scikit-learn
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c plotly plotly=4.14.3
pip install shapely kaggle simdkalman pytorch_lightning albumentations librosa segmentation-models-pytorch
pip install hydra-core --upgrade
# pip install 'neptune-client[pytorch-lightning]'

# lightgbm install
# sudo apt install libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-libopencl1 ocl-icd-opencl-de
# sudo apt install cmake
pip install lightgbm --install-option=--gpu

