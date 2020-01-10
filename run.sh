#!/bin/bash

echo "Installing Apex for 16-bit training (this makes training much faster)"
pip install -r requirements.txt

git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global_option="--cpp_ext" --global-option="--cuda_ext" ./apex

