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
```

### Hardware Requirements
- 5 Intel RealSense depth cameras (D435 or similar)
- USB 3.0 ports for camera connectivity
- A system with GPU support is recommended for YOLO detection

### Camera Configuration
Create a JSON configuration file `CAMERAS.json` with the serial numbers of the 5 RealSense cameras:

```json
{
    "serials": [
        "1234567890",
        "1234567891",
        "1234567892",
        "1234567893",
        "1234567894"
    ],
    "extrinsics": {
        "1234567890": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
        ...
    }
}
```

Each camera's extrinsic transformation matrix should be provided to align them in the 3D space.

## Usage
### Running Object Detection
To start object detection using 5 cameras:

```bash
python multidetector.py
```

This will:
- Initialize the RealSense cameras
- Run YOLO object detection on each camera
- Visualize the 2D detection results
- Optionally, display the detections in a 3D plot

### Merging Detections
To merge detections across multiple cameras and visualize the results:

```bash
python merger.py
```

This will:
- Start YOLO detection on all cameras
- Merge similar objects detected from different perspectives
- Display both individual and merged detections in a 2D plot and camera feed

### Arguments
You can modify the behavior of the detectors with the following arguments in `multidetector.py` and `merger.py`:
- `camera_config_path`: Path to the JSON file containing camera serials and extrinsics.
- `detection_type`: Choose between `'box'` for bounding box detection or `'mask'` for segmentation mask detection.
- `show`: Enable/disable showing the real-time video feed.
- `draw`: Enable/disable annotations on the video feed.
- `plot`: Enable/disable 3D plotting of object positions.
- `verbose`: Enable/disable verbose output for debugging.

Example:
```python
detector = MultiDetector(camera_config_path='CAMERAS.json', detection_type='box', show=True, draw=True, plot=False)
```

## File Overview
- `multidetector.py`: Main script that runs YOLO object detection on multiple RealSense cameras and outputs 3D coordinates for each detected object.
- `merger.py`: Script for merging object detections across cameras and plotting the results.
- `CAMERAS.json`: Configuration file specifying the serial numbers and extrinsics for each camera.

## Examples
### Real-Time Detection Example
![YOLO Object Detection in Surround View](https://via.placeholder.com/640x360)

The above image shows object detection results from multiple cameras stitched together in real-time.

### 3D Plot Example
![3D Object Tracking](https://via.placeholder.com/640x360)

The system converts the 2D object positions into 3D coordinates, which are displayed in a 3D plot.

## Contributions
Contributions are welcome! Please open a pull request or submit an issue for any bug reports, feature requests, or improvements.

