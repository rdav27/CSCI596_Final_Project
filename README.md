# CSCI596 Final Project

Implementation of a rea-time image object detector YOLO that supports parallel computing(CUDA) using PyTorch and OpenCV.

## Team Members

* Tianyue Fan
* Linhua Chen

## Dependencies

1. Python 3.5 and above
2. PyTorch 0.4 and above
3. OpenCV
4. [Pretrained weights file](https://pjreddie.com/media/files/yolov3.weights) for COCO dataset

## Executing Program

If your machine has CUDA enabled, the model will run on the GPU, which will be much faster than using CPU to make detections.
```
python detector.py --images <directory containing images> --det <directory to store detection results>
```
Other optional arguments includes:

`--bs` defines the batch size, default is 1

`--confidence` defines the object confidence to filter predictions, default is 0.5

`--nms_thresh` defines the NMS threshold, default is 0.4

`--cfg` defines the location of the configuration file, default is `cfg/yolov3.cfg`

`--weights` defines the location of the weights file, default is `yolov3.weights`

`--reso` defines the input resolution of the model, default is 416

## Detection Result Examples

## How Does the Detector Work

## Acknowledgments

* [The paper of YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* [Official code of YOLOv3](https://github.com/pjreddie/darknet)