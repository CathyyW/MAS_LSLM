cd /root/autodl-tmp/MAS_LSLM

conda create -n mas python=3.10 -y
conda activate mas

source /etc/network_turbo

pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

source env.sh