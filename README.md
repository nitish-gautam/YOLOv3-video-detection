# YOLOv3-video-detection
This AIM of this repository is to create  real time / video application using Deep Learning based Object Detection using YOLOv3 with OpenCV
YOLO trained on the COCO dataset. The COCO dataset consists of 80 labels.

### Dependencies
<ul>
    <li> 
        <a href="https://pjreddie.com/darknet/yolo/" >YOLO</a>
    </li>
    <li>
        <a href="https://opencv.org/" >OpenCV</a>
    </li>
</ul>

You also need to download the `yolo.weights` file and place it as described below :

You can download the weights by - 
```
    $ wget https://pjreddie.com/media/files/yolov3.weights
    or 
    https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

The directories structure should as follow:

yolo-coco-data/
 : The YOLOv3 object detector pre-trained (on the COCO dataset) model files. These were trained by the Darknet team should be kept here.

images/
 : This folder should contain static images which we will be used to perform object detection on for testing and evaluation purposes.

videos/
 : This directory should contains sample test videos for testing. After performing object detection with YOLO on video, we’ll process videos in real time camera input. Also Output videos that have been processed by YOLO and annotated with bounding boxes and class names will appear at this location.

### RESULT
![Capture3](https://user-images.githubusercontent.com/46977634/80386874-e0263780-889f-11ea-9db5-42aada0293cf.JPG)
