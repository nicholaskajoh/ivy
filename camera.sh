# RUN VEHICLE COUNTING SYSTEM FROM CAMERA INPUT USING A VIDEO FILE

# UBUNTU/LINUX SETUP

# 1. INSTALL v4l2loopback

# sudo apt-get install build-essential checkinstall
# wget https://github.com/umlaeute/v4l2loopback/archive/master.zip
# unzip master.zip
# cd v4l2loopback-master
# make
# sudo checkinstall --pkgname=v4l2loopback --pkgversion="$(date +%Y%m%d%H%M)-git" --default
# sudo make install
# sudo depmod -a
# sudo modprobe v4l2loopback

# 2. RUN THIS FILE (ENSURE VIDEO FILE AND DEVICE CAMERA IS CORRECTLY NAMED)

# chmod +x camera.sh
# ./camera.sh



video="$(pwd)/videos/sample_traffic_scene.mp4"
ffmpeg -re -i $video -r 1 -map 0:v -f v4l2 /dev/video1 & python3 Vehicle_Counting.py 1 --iscam --droi "750,400|1150,400|1850,700|1850,1050|500,1050" --showdroi

# TEST (PLAY VIDEO)
# ffplay /dev/video1