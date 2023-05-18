apt-get update && apt-get install libgl1

mkdir data
wget -N -P data https://pjreddie.com/media/files/yolov3.weights
wget -N -P data https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget -N -P data https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -N -P data https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# -- for pytorch
#wget -N https://raw.githubusercontent.com/x4nth055/pythoncode-tutorials/master/machine-learning/object-detection/utils.py
#wget -N https://raw.githubusercontent.com/x4nth055/pythoncode-tutorials/master/machine-learning/object-detection/darknet.py
#pip install torch

pip install opencv-contrib-python numpy matplotlib pafy imutils
pip install flask flask-login flask_wtf
pip install git+https://github.com/ytdl-org/youtube-dl.git@master#egg=youtube_dl

echo "patch pafy lib, Note Dir may vary"
cp backend_youtube_dl.py /usr/local/lib/python3.8/dist-packages/pafy/
