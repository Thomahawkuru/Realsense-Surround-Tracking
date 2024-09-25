import cv2
import numpy as np
import json
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
from FUNCTIONS import *  # Assuming this contains your custom functions

draw = True  # Set to True if you want to draw bounding boxes or masks on the frames
plot = True  # Set to True if you want to plot the 3D positions of detected objects
detection_type = 'box'  # Specify 'box' for bounding boxes or 'mask' for segmentation masks

# Load camera serials and extrinsics from CAMERAS.json
with open('CAMERAS.json', 'r') as f:
    camera_data = json.load(f)
    serials = camera_data["serials"]  # List of camera serial numbers
    extrinsics = camera_data.get("extrinsics", {})  # Camera extrinsic parameters

# Initialize RealSense pipeline for each camera
pipelines, aligns, profiles = initialize_cameras(serials=serials, resolution=[640, 360], fps=30)

# Initialize YOLO models based on the specified detection type
if detection_type == 'box':
    models = [YOLO("yolov10x.pt") for _ in range(len(pipelines))]  # Load YOLO model for bounding box detection
elif detection_type == 'mask':
    models = [YOLO("yolov9e-seg.pt") for _ in range(len(pipelines))]  # Load YOLO model for mask segmentation

# Prepare lists to store annotated frames and 3D coordinates for each camera
annotated_frames = [None] * len(pipelines)
robot_coordinates_list = [None] * len(pipelines)

# Initialize matplotlib for 3D plotting, if enabled
if plot:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-3, 3])  # Set limits for X-axis
    ax.set_ylim([-3, 3])   # Set limits for Y-axis
    ax.set_zlim([-1, 2])  # Set limits for Z-axis
    
    scatters = []  # List to hold scatter plot objects
    colors = ['r', 'b', 'g', 'y', 'k']  # Color options for different cameras
    labels = ['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4', 'Camera 5']  # Labels for legend
    # Create scatter plot objects that can be updated later
    for i in range(len(serials)):
        scatters.append(ax.scatter([], [], [], c=colors[i], marker='o', label=labels[i]))
    
    # Add legend to the plot
    ax.legend(loc='upper right')

# Define the camera processing function for threaded execution
def process_camera(pipeline, align, profile, model, idx, camera_transform):
    global annotated_frames, robot_coordinates_list, stop_threads
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()  # Get camera intrinsics

    while True:
        # Wait for frames from the camera
        frame = pipeline.wait_for_frames()

        # Align depth frame to color frame
        aligned_frame = align.process(frame)
        depth_frame = aligned_frame.get_depth_frame()  # Get depth frame
        color_frame = aligned_frame.get_color_frame()  # Get color frame

        # Check if any frames are missing
        if not depth_frame or not color_frame:
            continue  # Skip this iteration if frames are not available

        # Convert frames to numpy arrays for processing
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO tracking on the color image
        results = model.track(color_image, persist=True)

        # Visualize the results on the annotated frame
        if draw:
            if detection_type == 'box':
                annotated_frame = results[0].plot(boxes=True, masks=False)  # Draw bounding boxes
            elif detection_type == 'mask':
                annotated_frame = results[0].plot(boxes=False, masks=True)  # Draw segmentation masks
        else:
            annotated_frame = color_image.copy()  # Use the original color image if not drawing

        # Lists to store 3D coordinates and object IDs
        x_robot, y_robot, z_robot, object_ids = [], [], [], []

        # Loop through each detected object in the results
        for result in results:
            if detection_type == 'box':
                boxes = result.boxes
                classes = result.boxes.cls
                names = result.names
                if boxes is not None:
                    ids = boxes.id.tolist() if boxes.id is not None else []  # Get object IDs
                    for i, (box, obj_id, obj_class) in enumerate(zip(boxes.xyxy, ids, classes)):
                        obj_name = names[obj_class.item()]

                        # Extract the bounding box coordinates (x_min, y_min, x_max, y_max)
                        x_min, y_min, x_max, y_max = map(int, box)

                        # Calculate the center of the bounding box
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2

                        # Extract depth value at the center of the bounding box
                        depth_at_center = depth_image[center_y, center_x] * depth_frame.get_units()

                        # Convert depth and pixel coordinates to 3D coordinates
                        x = (center_x - intrinsics.ppx) * depth_at_center / intrinsics.fx
                        y = (center_y - intrinsics.ppy) * depth_at_center / intrinsics.fy
                        z = depth_at_center

                        # Transform to robot coordinates using the camera extrinsic matrix
                        robot_coordinates = np.dot(camera_transform, np.array([x, y, z, 1]))
                        robot_x, robot_y, robot_z = robot_coordinates[:3]

                        # Append robot coordinates and object ID to lists
                        x_robot.append(robot_x)
                        y_robot.append(robot_y)
                        z_robot.append(robot_z)
                        object_ids.append(f"ID {obj_id}")

                        if draw:
                            # Annotate the frame with the object's robot position
                            label = f"ID {obj_id} ({obj_name}):\n({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f})"
                            lines = label.split('\n')
                            for j, line in enumerate(lines):
                                text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                text_width, text_height = text_size
                                text_x = center_x - text_width // 2
                                text_y = center_y + (j * 3 + 1) * text_height // 2
                                cv2.putText(annotated_frame, line, (text_x, text_y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            elif detection_type == 'mask':  # For segmentation
                if result.masks:
                    masks = result.masks.data.cpu().numpy()  # Extract mask data
                    boxes = result.boxes
                    classes = result.boxes.cls
                    names = result.names

                    if boxes is not None:
                        ids = boxes.id.tolist() if boxes.id is not None else []  # Get object IDs
                        for i, (mask, obj_id, obj_class) in enumerate(zip(masks, ids, classes)):
                            obj_name = names[obj_class.item()]

                            # Get the mask indices for the segmented object
                            mask_indices = np.where(mask == 1)

                            # Clip indices to avoid out-of-bounds errors
                            mask_indices = (np.clip(mask_indices[0], 0, depth_image.shape[0] - 1),
                                            np.clip(mask_indices[1], 0, depth_image.shape[1] - 1))

                            # Extract depth values for the object using the mask
                            object_depth_values = depth_image[mask_indices] * depth_frame.get_units()

                            # Calculate the average depth of the object
                            average_depth = np.mean(object_depth_values) if object_depth_values.size > 0 else 0

                            # Calculate the centroid of the mask
                            if mask_indices[0].size > 0 and mask_indices[1].size > 0:
                                center_x = int(np.mean(mask_indices[1]))  # X coordinate (columns)
                                center_y = int(np.mean(mask_indices[0]))  # Y coordinate (rows)
                            else:
                                center_x, center_y = 0, 0

                            # Convert depth and pixel coordinates to 3D coordinates
                            x = (center_x - intrinsics.ppx) * average_depth / intrinsics.fx
                            y = (center_y - intrinsics.ppy) * average_depth / intrinsics.fy
                            z = average_depth

                            # Transform to robot coordinates using the camera extrinsic matrix
                            robot_coordinates = np.dot(camera_transform, np.array([x, y, z, 1]))
                            robot_x, robot_y, robot_z = robot_coordinates[:3]

                            # Append robot coordinates and object ID to lists
                            x_robot.append(robot_x)
                            y_robot.append(-robot_y)  # Negate y-coordinate for consistency
                            z_robot.append(robot_z)
                            object_ids.append(f"ID {obj_id}")

                            if draw:
                                # Annotate the frame with the object's robot position
                                label = f"ID {obj_id} ({obj_name}):\n({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f})"
                                lines = label.split('\n')
                                for j, line in enumerate(lines):
                                    text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                    text_width, text_height = text_size
                                    text_x = center_x - text_width // 2
                                    text_y = center_y + (j * 3 + 1) * text_height // 2
                                    cv2.putText(annotated_frame, line, (text_x, text_y), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Store the annotated frame and 3D coordinates for the camera
        annotated_frames[idx] = annotated_frame
        robot_coordinates_list[idx] = (x_robot, z_robot, y_robot, object_ids)

        if stop_threads:
            break  # Exit the loop if stop signal is set

# Create and start a thread for each camera
threads = []
stop_threads = False
for i, (pipeline, align, profile, model, serial) in enumerate(zip(pipelines, aligns, profiles, models, serials)):
    camera_transform = np.array(extrinsics.get(serial, np.eye(4)))  # Default to identity matrix if extrinsics not provided
    thread = threading.Thread(target=process_camera, args=(pipeline, align, profile, model, i, camera_transform))
    thread.start()  # Start the camera processing thread
    threads.append(thread)  # Append thread to the list

# Display the combined results from all cameras
try:
    while True:
        # Combine the annotated frames side by side (when available)
        if all(frame is not None for frame in annotated_frames) and draw:
            combined_frame = np.hstack(annotated_frames)  # Horizontally stack annotated frames
            cv2.imshow("YOLO Detection with Depth and Masks", combined_frame)

        # Update the 3D plot with robot coordinates from each camera
        if plot and all(coords is not None for coords in robot_coordinates_list):
            for (scatter, coords) in zip(scatters, robot_coordinates_list):
                scatter._offsets3d = coords[:3]  # Update scatter plot data with robot coordinates
                plt.draw()  # Redraw the plot
                plt.pause(0.001)  # Pause to update the plot

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Stop threads and release resources before exiting
    stop_threads = True  # Signal threads to stop
    for thread in threads:
        thread.join()  # Wait for all threads to finish
    for pipeline in pipelines:
        pipeline.stop()  # Stop the camera pipelines
    cv2.destroyAllWindows()  # Close all OpenCV windows
