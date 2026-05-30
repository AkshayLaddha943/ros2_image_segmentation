# Image Segmentation
This is a Pytorch adaptation of the Image Segmentation assignment from Automated and Connected Driving Challenges course of edX. The pipeline applies pre-trained segmentation models on actual camera feeds in rosbag dataset. 

Get the rosbag file - https://drive.google.com/drive/folders/1kisTi85pfxdFahnEdTVBe9vxXGRJKUhg

## Overview -
This repo implements the following features:
* Loading the rosbag file
* Resizing and applying batch dimension to the image frames
* Convert segmentation map to RGB encoding (H,W) -> (H, W, 3)
* Load the pre-trained model semantic segmentation from torchvision package
* Visualize the segmented output in Rviz

For Segmentation, we use the DeepLabV3 model with Resnet101 as backbone which has been trained on a subset of COCO 2017 dataset corresponding to PascalVOC with 20 supported classes

## Installation -
* Clone the Repository
```
git clone https://github.com/AkshayLaddha943/ros2_image_segmentation
```
* Build the Package
```
colcon build --symlink install
```
* Source the workspace
```
source install/setup.bash
```
* Run the rosbag and the script in separate terminals
```
ros2 bag play course_ws/src/image_segmentation/bags/left_camera_templergraben.db3
ros2 run image_segmenation im_segmentation
```


## Results - 


https://github.com/user-attachments/assets/5a5150d1-4cd2-45d3-b3b9-f71e1b5ab685

