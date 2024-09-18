from collections import defaultdict
import cv2,json
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from FUNCTIONS import *

draw = True
plot = False  # Set to True if you want to plot the 3D positions

# Initialize YOLOv9 segmentation model
model = YOLO("yolov9e-seg.pt")

# Initialize RealSense pipeline for cameras
with open('CAMERAS.json', 'r') as f:
    pipelines, aligns = initialize_cameras(serials=json.load(f).get("serials", []))

# Define camera intrinsic parameters (adjust based on your calibration)
fx, fy = 632, 630  # Focal lengths in pixels
cx, cy = 636, 363  # Principal point in pixels (center)

# Define camera extrinsic parameters (transformation matrix from camera frame to robot frame)
camera_to_robot_transform = np.array([
    [-0.05030855, -0.99862953, -0.01442574, 0.1],
    [0.27563736, 0, -0.9612617, -0.0475],
    [0.95994432, -0.05233596, 0.2752596, -0.0485],
    [0, 0, 0, 1]
], dtype=np.float32)

# Function to crop an image
crop_per_side_pixels = 200

# Processing loop
try:
    while True:
        # Wait for RealSense frames from both cameras
        frames = []
        for pipeline, align in zip(pipelines, aligns):
            frame = pipeline.wait_for_frames()
            frames.append(frame)
        
        # Align depth to color for each camera
        aligned_frames = [align.process(frame) for align, frame in zip(aligns, frames)]
        
        # Extract depth and color frames for each camera
        depth_frames = [aligned_frame.get_depth_frame() for aligned_frame in aligned_frames]
        color_frames = [aligned_frame.get_color_frame() for aligned_frame in aligned_frames]

        # Check if any of the frames are missing
        if any(frame is None for frame in depth_frames + color_frames):
            continue

        # Convert frames to numpy arrays
        depth_images = [np.asanyarray(depth_frame.get_data()) for depth_frame in depth_frames]
        color_images = [np.asanyarray(color_frame.get_data()) for color_frame in color_frames]

        # Rectify each color image
        rectified_colors = [rectify_image(color_image, tilt_angle=15) for color_image in color_images]

        # Crop each rectified image
        cropped_colors = [crop_image(rectified_color, crop_per_side_pixels) for rectified_color in rectified_colors]
        cropped_depths = [crop_image(depth_image, crop_per_side_pixels) for depth_image in depth_images]

        # Stitch the cropped images horizontally
        stitched_color_image = np.hstack(cropped_colors)
        stitched_depth_image = np.hstack(cropped_depths)
        
        # Run YOLOv9 tracking on the stitched color image
        results = model.track(stitched_color_image, persist=True)

        # Visualize the results on the frame (draw bounding boxes and masks)
        if draw:
            annotated_frame = results[0].plot(boxes=False, masks=True)

        # Lists to store 3D coordinates and object IDs
        x_robot = []
        y_robot = []
        z_robot = []
        object_ids = []

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

                        # Upscale the mask from half resolution to full resolution
                        mask_upscaled = cv2.resize(mask, (stitched_depth_image.shape[1], stitched_depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Get the mask indices where the mask is active (i.e., the object area)
                        mask_indices = np.where(mask_upscaled == 1)

                        # Extract depth values for the object using the upscaled mask
                        object_depth_values = remove_outliers(stitched_depth_image[mask_indices])

                        # Calculate the average depth of the object
                        if object_depth_values.size > 0:
                            average_depth = np.mean(object_depth_values) * depth_frames[0].get_units()
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
                        y = (center_y - cy) * average_depth / fy
                        z = average_depth

                        # Transform to robot coordinates
                        robot_coordinates = np.dot(camera_to_robot_transform, np.array([x, y, z, 1]))
                        robot_x, robot_y, robot_z = robot_coordinates[:3]

                        # Append to lists
                        x_robot.append(robot_z)
                        y_robot.append(-robot_y)
                        z_robot.append(robot_x)

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

        # Update the 3D plot
        if plot: update_plot(x_robot, y_robot, z_robot, object_ids)

        # Display the annotated frame with bounding boxes and masks
        if draw: cv2.imshow("YOLOv9 Detection with Depth and Masks", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()