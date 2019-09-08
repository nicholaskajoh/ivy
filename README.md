# Video-based Vehicle Counting System
![](vehicle_counting.png)

## Setup
- Install Python 3 on your machine.
- Clone this repo `git@github.com:nicholaskajoh/Vehicle-Counting.git`.
- Get video footage of a traffic scene ([sample videos](https://drive.google.com/drive/folders/1h8ANowkfm4TXGDg7R5Z6rnosySVc-Ht7?usp=sharing)).
- Create and/or use a virtual environment.
- Run `pip install -r requirements.txt` to install dependencies.

## Run

### Configuration
```
usage: main.py [-h] [--iscam] [--droi DROI] [--showdroi] [--mcdf MCDF]
               [--mctf MCTF] [--di DI] [--detector DETECTOR]
               [--tracker TRACKER] [--record] [--headless]
               [--clposition CLPOSITION]
               video

positional arguments:
  video                 relative/absolute path to video or camera input of
                        traffic scene

optional arguments:
  -h, --help            show this help message and exit
  --iscam               specify if video capture is from a camera
  --droi DROI           specify a detection region of interest (ROI) i.e a set
                        of vertices that represent the area (polygon) where
                        you want detections to be made (format:
                        1,2|3,4|5,6|7,8|9,10 default: 0,0|frame_width,0|frame_
                        width,frame_height|0,frame_height [i.e the whole video
                        frame])
  --showdroi            display/overlay the detection roi on the video
  --mcdf MCDF           maximum consecutive detection failures i.e number of
                        detection failures before it's concluded that an
                        object is no longer in the frame
  --mctf MCTF           maximum consecutive tracking failures i.e number of
                        tracking failures before the tracker concludes the
                        tracked object has left the frame
  --di DI               detection interval i.e number of frames before
                        detection is carried out again (in order to find new
                        vehicles and update the trackers of old ones)
  --detector DETECTOR   select a model/algorithm to use for vehicle detection
                        (options: yolo, haarc, bgsub, ssd, tfoda | default:
                        yolo)
  --tracker TRACKER     select a model/algorithm to use for vehicle tracking
                        (options: csrt, kcf, camshift | default: kcf)
  --record              record video and vehicle count logs
  --headless            run VCS without UI display
  --clposition CLPOSITION
                        position of counting line (options: top, bottom, left,
                        right | default: bottom)
```

### Notes
- To use the `yolo` detector (i.e You Only Look Once neural net), copy [detectors/yolo/.env.example](/detectors/yolo/.env.example) to detectors/yolo/.env and edit as appropriate. You can try out this detector [with these pre-trained models](https://pjreddie.com/darknet/yolo/).
- To use the `tfoda` detector (i.e Tensorflow Object Detection API), copy [detectors/tfoda/.env.example](/detectors/tfoda/.env.example) to detectors/tfoda/.env and edit as appropriate. You can try out this detector [with these pre-trained models](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#use-existing-config-file-for-your-model).
- To use the `ssd` detector, download this [pre-trained model](https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view) and place it in the [detectors/ssd folder](/detectors/ssd).

### Examples
Use defaults:

```
python main.py "./data/videos/sample_traffic_scene.mp4"
```

Custom configuration:

```
python main.py "./data/videos/sample_traffic_scene.mp4" --droi "750,400|1150,400|1850,700|1850,1050|500,1050" --showdroi --detector "haarc" --tracker "csrt" --di 5 --mctf 15
```

With camera input:

```
python main.py 1 --iscam
```

__NB:__ You can press the `s` key when the program is running to capture a screenshot. The images are saved in the [screenshots folder](/screenshots).

## How it works
The vehicle counting system is made up of three main components: a detector, tracker and counter. The detector identifies vehicles in a given frame of video and returns a list of bounding boxes around the vehicles to the tracker. The tracker uses the bounding boxes to track the vehicles in subsequent frames. The detector is also used to update the trackers periodically to ensure that they are still tracking the vehicles correctly. The counter counts vehicles when they leave the frame or makes use of a counting line drawn across a road.

__PS:__ You can find out about how the vehicle counting system was built by checking out this article on my blog: https://alphacoder.xyz/vehicle-counting/.

## Donate
Love this project? You can [buy me a coffee or two](http://buymeacoff.ee/nicholaskajoh) to support its continued development. 😊