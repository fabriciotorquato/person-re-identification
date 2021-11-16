conda init bash

conda create -n tcc-25 python=3.6 anaconda
conda activate tcc-25
pip install -r requirements.txt 
conda install -c anaconda cudatoolkit==11.2 -y
conda install -c anaconda cudnn -y



conda create -n tcc-17 python=3.6 anaconda
conda activate tcc-17
conda install -c anaconda cudatoolkit==9.0 -y
conda install -c anaconda cudnn -y
pip install -r requirements_facenet.txt
conda deactivate