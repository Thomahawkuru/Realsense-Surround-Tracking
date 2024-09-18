import cv2
import json
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
from FUNCTIONS import *

draw = True
plot = False  # Set to True if you want to plot the 3D positions

# Initialize RealSense pipeline for cameras
with open('CAMERAS.json', 'r') as f:
    camera_config = json.load(f)
    pipelines, aligns = initialize_cameras(serials=camera_config.get("serials", []), resolution=[640, 360])

# Define camera intrinsic parameters (adjust based on your calibration)
fx, fy = 640, 640  # Focal lengths in pixels
cx, cy = 640, 360  # Principal point in pixels (center)

# Define camera extrinsic parameters (transformation matrix from camera frame to robot frame)
camera_transform = np.array([
    [1, 0, 0, 0],  # No rotation on X, no translation on X
    [0, 1, 0, 0],  # No rotation on Y, no translation on Y
    [0, 0, 1, 0],  # No rotation on Z, no translation on Z
    [0, 0, 0, 1]   # Homogeneous coordinate
], dtype=np.float32)

# Initialize a YOLOv9 model for each camera
# Initialize a YOLOv9 model for each camera
models = [YOLO("yolov9e-seg.pt") for _ in range(len(pipelines))]

# Store annotated frames and 3D coordinates for each camera
annotated_frames = [None] * len(pipelines)
robot_coordinates_list = [None] * len(pipelines)

# Camera processing function for threading
def process_camera(pipeline, align, model, idx):
    global annotated_frames, robot_coordinates_list

    while True:
        # Wait for frames
        frame = pipeline.wait_for_frames()

        # Align depth to color
        aligned_frame = align.process(frame)
        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()

        # Check if any frames are missing
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv9 tracking on the color image
        results = model.track(color_image, persist=True)

        # Visualize the results on the frame (draw bounding boxes and masks)
        if draw:
            annotated_frame = results[0].plot(boxes=False, masks=True)

        # Lists to store 3D coordinates and object IDs
        x_robot, y_robot, z_robot, object_ids = [], [], [], []

        # Loop through each detected object
        for result in results:
            if result.masks:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                classes = result.boxes.cls
                names = result.names

                if boxes is not None:
                    ids = boxes.id.tolist() if boxes.id is not None else []

                    # Loop over all detected objects in the current frame
                    for i, (mask, obj_id, obj_class) in enumerate(zip(masks, ids, classes)):
                        obj_name = names[obj_class.item()]

                        # Get the mask indices where the mask is active (i.e., the object area)
                        mask_indices = np.where(mask == 1)

                        # Clip indices to avoid out-of-bounds errors
                        mask_indices = (np.clip(mask_indices[0], 0, depth_image.shape[0] - 1),
                                        np.clip(mask_indices[1], 0, depth_image.shape[1] - 1))

                        # Extract depth values for the object using the mask
                        object_depth_values = remove_outliers(depth_image[mask_indices])

                        # Calculate the average depth of the object
                        if object_depth_values.size > 0:
                            average_depth = np.mean(object_depth_values) * depth_frame.get_units()
                        else:
                            average_depth = 0

                        # Calculate the centroid of the mask
                        if mask_indices[0].size > 0 and mask_indices[1].size > 0:
                            center_x = int(np.mean(mask_indices[1]))  # X coordinate (columns)
                            center_y = int(np.mean(mask_indices[0]))  # Y coordinate (rows)
                        else:
                            center_x, center_y = 0, 0

                        # Convert depth and pixel coordinates to 3D robot coordinates
                        x = (center_x - cx) * average_depth / fx
                        y = -(center_y - cy) * average_depth / fy
                        z = average_depth

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
                            # Annotate the frame with the object's robot position at the centroid of the mask
                            label = f"ID {obj_id} ({obj_name}):\n({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f})"
                            
                            # Split the label into multiple lines
                            lines = label.split('\n')
                            for j, line in enumerate(lines):
                                # Calculate text size and adjust position for centering
                                text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                text_width, text_height = text_size
                                text_x = center_x - text_width // 2
                                text_y = center_y + (j*3 + 1) * text_height // 2

                                # Use cv2.putText to add robot position information at the center of the mask
                                cv2.putText(annotated_frame, line, (text_x, text_y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Store the annotated frame and 3D coordinates for the camera
        annotated_frames[idx] = annotated_frame
        robot_coordinates_list[idx] = (x_robot, y_robot, z_robot, object_ids)

# Create and start a thread for each camera
threads = []
for i, (pipeline, align, model) in enumerate(zip(pipelines, aligns, models)):
    thread = threading.Thread(target=process_camera, args=(pipeline, align, model, i))
    threads.append(thread)
    thread.start()

# Display the combined results
try:
    while True:
        # Combine the annotated frames side by side (when available)
        if all(frame is not None for frame in annotated_frames):
            combined_frame = np.hstack(annotated_frames)
            cv2.imshow("YOLOv9 Detection with Depth and Masks", combined_frame)

        # Update the 3D plot with robot coordinates from each camera
        if plot and all(coords is not None for coords in robot_coordinates_list):
            all_x_robot, all_y_robot, all_z_robot, all_object_ids = [], [], [], []
            for coords in robot_coordinates_list:
                x_robot, y_robot, z_robot, object_ids = coords
                all_x_robot.extend(x_robot)
                all_y_robot.extend(y_robot)
                all_z_robot.extend(z_robot)
                all_object_ids.extend(object_ids)
            update_plot(all_x_robot, all_y_robot, all_z_robot, all_object_ids)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Stop threads and release resources
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
