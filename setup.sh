#!/bin/bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install matplotlib seaborn
conda install -c conda-forge opencv

pip install gymnasium\[all\]
pip install gymnasium\[accept-rom-license\]