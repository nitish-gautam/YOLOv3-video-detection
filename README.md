# YOLOv3-video-detection

YOLO trained on the COCO dataset. The COCO dataset consists of 80 labels. You can fetch the weights from the link below 
https://github.com/pjreddie/darknet/blob/master/data/coco.names

https://pjreddie.com/darknet/yolo/

The directories structure should as follow:

yolo-coco-data/
 : The YOLOv3 object detector pre-trained (on the COCO dataset) model files. These were trained by the Darknet team should be kept here.

images/
 : This folder should contain static images which we will be used to perform object detection on for testing and evaluation purposes.

videos/
 : This directory should contains sample test videos for testing. After performing object detection with YOLO on video, we’ll process videos in real time camera input. Also Output videos that have been processed by YOLO and annotated with bounding boxes and class names will appear at this location.

# RESULT
![Capture3](https://user-images.githubusercontent.com/46977634/80386874-e0263780-889f-11ea-9db5-42aada0293cf.JPG)
