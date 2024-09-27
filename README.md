# YOLO Object Tracking with multiple RealSense Cameras in Surround Configuration

This project implements real-time object detection and tracking using multiple Intel RealSense cameras and YOLO object detection models. The cameras are placed in a surround configuration to detect objects from multiple perspectives, calculate their 3D coordinates, and merge similar detections across the cameras. This system is meant for robotics and autonomous systems that require a 360-degree field of view.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [File Overview](#file-overview)

## Features
- **Multi-camera setup:** Uses multiple RealSense cameras for complete surround detection.
- **YOLOv10 and YOLOv9_seg models:** Support for both box-based and mask-based detection.
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
- one or multiple Intel RealSense depth cameras (D455 or similar)
- USB 3.0 ports for camera connectivity
- A system with CUDA support is recommended for YOLO detection

### Camera Configuration
Create a JSON configuration file `CAMERAS.json` with the serial numbers of the RealSense cameras:

```json
{
    "serials": [
        "1234567890",
        ...
    ],
    "extrinsics": {
        "1234567890": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
        ...
    }
}
```

Each camera's extrinsic transformation matrix should be provided to align them in the 3D space. 
The configuration can be verified using `plot_cameras.py`

## Usage
### Running Object Detection
To start object detection using 5 cameras:

```bash
python multidetector.py
```

This will:
- Initialize the RealSense cameras
- Run YOLO object detection on each camera in separate threads
- Optionally, Visualize the 2D detection results
- Optionally, display the detections in a 3D plot

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
detector = MultiDetector(camera_config_path='CAMERAS.json', detection_type='box', show=True, draw=True, plot=False, verbose=False)
detector.start()
```

### Merging Detections
To merge detections across multiple cameras and visualize the results:

```python
merger = Merger(camera_config_path='CAMERAS_Jackal04.json', threshold=0.35, show=True, plot=True)
merger.merge_and_plot_detections()
```

This will:
- Start YOLO detection on all cameras in separate threads using the MultiDetector class
- Merge similar objects detected from different perspectives
- Optionally, display both individual and merged detections in a 2D plot and camera feed


## File Overview
- `multidetector.py`: Main script that runs YOLO object detection on multiple RealSense cameras and outputs 3D coordinates for each detected object.
- `merger.py`: Script for merging object detections across cameras and plotting the results.
- `CAMERAS.json`: Configuration file specifying the serial numbers and extrinsics for each camera.
- `plot_cameras.py`: Visualize camera extrinsics specified in CAMERAS.json as verification

