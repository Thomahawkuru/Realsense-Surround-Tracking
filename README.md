# YOLO Object Tracking with 5 RealSense Cameras in Surround Configuration

This project implements real-time object detection and tracking using 5 Intel RealSense cameras and the YOLO (You Only Look Once) object detection model. The cameras are placed in a surround configuration to detect objects from multiple perspectives, calculate their 3D coordinates, and merge similar detections across the cameras. This system is ideal for robotics and autonomous systems that require a 360-degree field of view.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [File Overview](#file-overview)
- [Examples](#examples)
- [Contributions](#contributions)

## Features
- **Multi-camera setup:** Uses 5 RealSense cameras for complete surround detection.
- **YOLOv5 and YOLOv8 models:** Support for both box-based and mask-based detection.
- **Real-time 3D tracking:** Converts 2D detections into 3D coordinates using camera depth data.
- **Object merging:** Detects and merges similar objects across different camera views.
- **Live visualization:** Real-time display of detection results in both 2D and 3D.
  
## Setup
### Prerequisites
- Python 3.x
- Intel RealSense SDK (`pyrealsense2`)
- OpenCV
- YOLO models from Ultralytics
- NumPy
- Matplotlib

You can install the required dependencies using the following command:

```bash
pip install opencv-python numpy matplotlib pyrealsense2 ultralytics
