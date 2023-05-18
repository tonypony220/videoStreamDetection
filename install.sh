mkdir data
wget -N -P data https://pjreddie.com/media/files/yolov3.weights
wget -N -P data https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
#wget -N -P data https://pjreddie.com/media/files/classes.txt
wget -N -P data https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -N -P data https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

wget -N https://raw.githubusercontent.com/x4nth055/pythoncode-tutorials/master/machine-learning/object-detection/utils.py
wget -N https://raw.githubusercontent.com/x4nth055/pythoncode-tutorials/master/machine-learning/object-detection/darknet.py

pip3 install opencv-python numpy matplotlib
pip install torch
