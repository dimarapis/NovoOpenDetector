# SPOT DETECTION MODULE

## ONLY TESTED ON UBUNTU 20.04 with CONDA
### Open-set detection and segmentation module originally designed for spot robot picking operation. Given an image, it returns the bounding boxes of the objects of interest, segments the region of the object within the bounding box, returns a segmentation mask and a central point of the object to be used for the picking operation.

## Installation 
### Go to your project folder and clone repository
```
git clone https://github.com/dimarapis/NovoOpenDetector.git
```

### Navigate inside the cloned repository +folder
```
cd NovoOpenDetector
```

### Create virtual environment
Start by creating a virtual environment for your project, with python 3.10 (other version would work but this is the only I have tested for now)
```
conda create -n spot_detection python=3.10
conda activate spot_detection
```

### Run bash file that handles all downloads and package requirements
```
chmod +x setup_env.sh
./setup_env.sh
```

### Run script 
```
python demo.py 
```
The script will return detections, and return center points in the camera x,y. Also it will save an image at the results folder

# PROBLEMS WITH INSTALLATION
Sometimes the nvidia-cublas-cu11 installed with torch will introduce problems so you might need to uninstall with pip uninstall nvidia-clublas-cu11
 
## NOT IMPLEMENTED FOR NOW
### Optionally, you can give items as arguments. The model without arguments will look for "banana" and "bottle". 
