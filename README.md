## Installation

### Code installation

#### Install with conda and pip
```shell
# 1. Create anaconda environment
conda create -n vto-full python=3.10
conda activate vto-full

# 2. Install requirement
pip install -r requirement.txt

# 3. Install PyTorch
conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# 4. Install Pytorch3D
conda install -y -c iopath iopath
conda install -y -c bottler nvidiacub
conda install -y pytorch3d -c pytorch3d

### Models

#download smplx.zip
cd body_head_recovery/models
gdown https://drive.google.com/uc?id=1roylvU8XdxHuxelvzXYzx4L5zakLKXyP
unzip smplx.zip
rm -f smplx.zip
cd ../../

# Color transfer
cd body_head_recovery/Color_Transfer
gdown https://drive.google.com/uc?id=1ROs06448Vrpuaq6rwTgjTwrPhxKAJIry
unzip checkpoints.zip
rm -f checkpoints.zip
cd ../../

# Inpaiting
cd body_head_recovery/Inpainting/experiments
gdown https://drive.google.com/uc?id=1QNtqAYPVBr2eRx2GS1s7YjacXBQwNV5V
unzip celebahq.zip
rm -f celebahq.zip

## Running

python main_test.py"# body_head_recovery" 
# try-on-ai-ml
