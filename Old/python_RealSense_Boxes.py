from collections import defaultdict
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Old.FUNCTIONS import *

draw = True
plot = False  # Set to True if you want to plot the 3D positions

# Initialize YOLOv9 model
model = YOLO("yolov10x.pt")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640*2, 360*2, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640*2, 360*2, rs.format.bgr8, 30)
pipeline.start(config)

# Create align object to align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Define camera intrinsic parameters (adjust based on your calibration)
fx, fy = 605.5, 605.5  # Focal lengths in pixels
cx, cy = 640, 360  # Principal point in pixels (center)

# Define camera extrinsic parameters (transformation matrix from camera frame to robot frame)
camera_transform = np.array([
    [1, 0, 0, 0],  # No rotation on X, no translation on X
    [0, 1, 0, 0],  # No rotation on Y, no translation on Y
    [0, 0, 1, 0],  # No rotation on Z, no translation on Z
    [0, 0, 0, 1]   # Homogeneous coordinate
], dtype=np.float32)

# Processing loop
try:
    while True:
        # Wait for RealSense frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv9 tracking on the color image
        results = model.track(color_image, persist=True)

        # Visualize the results on the frame (draw bounding boxes)
        if draw:
            annotated_frame = results[0].plot(boxes=True, masks=False)

        # Lists to store 3D coordinates and object IDs
        x_robot = []
        y_robot = []
        z_robot = []
        object_ids = []

        # Loop through each detected object
        for result in results:
            boxes = result.boxes
            classes = result.boxes.cls
            names = result.names
            if boxes is not None:
                ids = boxes.id.tolist() if boxes.id is not None else []

                # Loop over all detected objects in the current frame
                for i, (box, obj_id, obj_class) in enumerate(zip(boxes.xyxy, ids, classes)):
                    obj_name = names[obj_class.item()]

                    # Extract the bounding box coordinates (x_min, y_min, x_max, y_max)
                    x_min, y_min, x_max, y_max = map(int, box)

                    # Calculate the center of the bounding box
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    # Extract depth value at the center of the bounding box
                    depth_at_center = depth_image[center_y, center_x] * depth_frame.get_units()

                    # Convert depth and pixel coordinates to 3D robot coordinates
                    x = (center_x - cx) * depth_at_center / fx
                    y = -(center_y - cy) * depth_at_center / fy
                    z = depth_at_center

                    # Transform to robot coordinates
                    robot_coordinates = np.dot(camera_transform, np.array([x, y, z, 1]))
                    robot_x, robot_y, robot_z = robot_coordinates[:3]

                    # Append to lists
                    x_robot.append(robot_x)
                    y_robot.append(-robot_y)
                    z_robot.append(robot_z)

                    # Append object ID and name to the list
                    object_ids.append(f"ID {obj_id}")

                    if draw:
                        # Annotate the frame with the object's robot position at the center of the bounding box
                        label = f"ID {obj_id} ({obj_name}):\n({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f})"
                        
                        # Split the label into multiple lines
                        lines = label.split('\n')
                        for j, line in enumerate(lines):
                            # Calculate text size and adjust position for centering
                            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            text_width, text_height = text_size
                            text_x = center_x - text_width // 2
                            text_y = center_y + (j*3 + 1) * text_height // 2

                            # Use cv2.putText to add robot position information at the center of the bounding box
                            cv2.putText(annotated_frame, line, (text_x, text_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Update the 3D plot
        if plot:
            update_plot(x_robot, y_robot, z_robot, object_ids)

        # Display the annotated frame with bounding boxes
        if draw:
            cv2.imshow("YOLOv10 Detection with Robot Position and Bounding Boxes", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    pipeline.stop()
    plt.close()
    cv2.destroyAllWindows()
