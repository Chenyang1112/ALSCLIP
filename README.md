# ALSCLIP-ALS point cloud segmentation

## 1. Requirements
Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.9.12, PyTorch 1.11.0, CUDA 11.3 and RTX 3090.
***
First, it is recommended to create a new environment and install PyTorch and torchvision. Next, please use the following command for installation.

```
pip install -r requirements.txt

# Install pointops2
cd lib/pointops2/
python setup.py install

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

In addition, you need to install CLIP according to [CLIP](https://github.com/openai/CLIP).

## 2. Datasets

We use DFC2019 dataset in this work. You can download the data from https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019

## 4. Usage
Training on DFC2019, run:
```
cd scripts
bash train.sh
```
Testing on DFC2019, run:
```
cd scripts
bash test.sh
```

## Acknowledgements

The code for this task is built upon [PointMixer](https://github.com/LifeBeyondExpectations/ECCV22-PointMixer).
