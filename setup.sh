#!/bin/bash
conda install -y pytorch::pytorch torchvision torchaudio -c pytorch
conda install -y matplotlib seaborn
conda install -y opencv -c conda-forge

pip install swig
pip install gymnasium\[all\]
pip install gymnasium\[accept-rom-license\]
pip install imageio