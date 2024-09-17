from collections import defaultdict
import cv2
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
camera_to_robot_transform = np.array([
    [-0.05030855, -0.99862953, -0.01442574, 0.1],
    [0.27563736, 0, -0.9612617, -0.0475],
    [0.95994432, -0.05233596, 0.2752596, -0.0485],
    [0, 0, 0, 1]
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
        probs = results[0].probs
        print(probs)
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
                        mask_upscaled = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Get the mask indices where the mask is active (i.e., the object area)
                        mask_indices = np.where(mask_upscaled == 1)

                        # Extract depth values for the object using the upscaled mask
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
        if plot:
            update_plot(x_robot, y_robot, z_robot, object_ids)

        # Display the annotated frame with masks
        if draw:
            cv2.imshow("YOLOv9 Detection with Robot Position and Masks", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    pipeline.stop()
    plt.close()
    cv2.destroyAllWindows()
