conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the env for grounded-sam-2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2/checkpoints/
bash download_ckpts.sh
cd ../gdino_checkpoints/
bash download_ckpts.sh
cd ../
pip install -e .
pip install --no-build-isolation -e grounding_dino

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

pip install gsplat==1.4.0
pip install kornia
cd gaussian_splatting/submodules/diff-gaussian-rasterization/
python setup.py build_ext --inplace
pip install -e .
cd ../simple-knn/
pip install -e .
cd ../../../

pip install plyfile

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d/
pip install -e .
cd ../

pip install einops

# Install the env for trellis
cd data_process
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
cd ../..

wget https://github.com/Zarrac/flashattention-blackwell-wheels-whl-ONLY-5090-5080-5070-5060-flash-attention-/releases/download/FlashAttention/flash_attn-2.7.4.post1-rtx5090-torch2.7.0cu128cxx11abiTRUE-cp310-linux_x86_64.whl
mv flash_attn-2.7.4.post1-rtx5090-torch2.7.0cu128cxx11abiTRUE-cp310-linux_x86_64.whl flash_attn-2.7.4.post1-0rtx5090torch270cu128cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1-0rtx5090torch270cu128cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1-0rtx5090torch270cu128cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
