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
usage: Vehicle_Counting.py [-h] [--iscam] [--droi DROI] [--showdroi]
                           [--mcdf MCDF] [--mctf MCTF] [--di DI]
                           [--detector DETECTOR] [--tracker TRACKER]
                           [--record] [--headless] [--clposition CLPOSITION]
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
                        (options: yolo, haarc, bgsub, ssd | default: yolo)
  --tracker TRACKER     select a model/algorithm to use for vehicle tracking
                        (options: csrt, kcf, camshift | default: kcf)
  --record              record video and vehicle count logs
  --headless            run VCS without UI display
  --clposition CLPOSITION
                        position of counting line (options: top, bottom, left,
                        right | default: bottom)
```

### Notes
- To use the `yolo` detector, download the [YOLO v3 weights](https://pjreddie.com/media/files/yolov3.weights) and place it in the [detectors/yolo folder](/detectors/yolo).
- To use the `ssd` detector, download this [pre-trained model](https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view) and place it in the [detectors/ssd folder](/detectors/ssd).

### Examples
Use defaults:

```
python Vehicle_Counting.py "./videos/sample_traffic_scene.mp4"
```

Custom configuration:

```
python Vehicle_Counting.py "./videos/sample_traffic_scene.mp4" --droi "750,400|1150,400|1850,700|1850,1050|500,1050" --showdroi --detector "haarc" --tracker "csrt" --di 5 --mctf 15
```

With camera input:

```
python Vehicle_Counting.py 1 --iscam
```

__NB:__ You can press the `s` key when the program is running to capture a screenshot. The images are saved in the [screenshots folder](/screenshots).

## How it works
The vehicle counting system is made up of three main components: a detector, tracker and counter. The detector identifies vehicles in a given frame of video and returns a list of bounding boxes around the vehicles to the tracker. The tracker uses the bounding boxes to track the vehicles in subsequent frames. The detector is also used to update trackers periodically to ensure that they are still tracking the vehicles correctly. The counter draws a counting lines across the road. When a vehicle crosses the line, the vehicle count is incremented.