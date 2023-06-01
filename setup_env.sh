echo "Setting up environment requirements" 

#Setup script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#Install all the required packages for the project
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -q -e .

#Supervision library for annotations, can be removed
pip uninstall -y supervision
pip install -q supervision==0.6.0

#Download weights for the model
cd $SCRIPT_DIR/weights
#https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd $SCRIPT_DIR/data
#Download sample data
gdown --folder https://drive.google.com/drive/folders/1z6juUm46GeMcJqLvvYy09FUAVA6YAn-y

cd $SCRIPT_DIR
