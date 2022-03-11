#!/usr/bin/env bash

apt-get install -y python python-pip python-dev

pip install subprocess32 gradescope-utils

apt-get update

apt-get install -y software-properties-common python-software-properties

add-apt-repository ppa:deadsnakes/ppa

apt-get update

apt update

apt install -y python3.6

apt install -y python3.6-dev

apt install -y python3.6-venv

apt-get update
apt-get install -y python3-tk

apt-get install -y python3-pip

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade scipy

# Builds fail when the final output is to std.err (including non-fatal warnings), include an echo:
echo "Build successful <:)"