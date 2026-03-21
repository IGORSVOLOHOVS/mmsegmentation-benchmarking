python3.10 -m venv venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
python -m mim install mmengine
python -m mim install "mmcv==2.1.0"

git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
pip install ftfy==6.3.1
pip install regex==2025.9.18
pip install numpy==1.26.4
pip install clearml==2.0.2

cd mmsegmentation
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg