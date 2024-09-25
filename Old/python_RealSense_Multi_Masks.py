import cv2
import json
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
from Old.FUNCTIONS import *

draw = True
plot = True  # Set to True if you want to plot the 3D positions

# Load serials and extrinsics from CAMERAS.json
with open('CAMERAS.json', 'r') as f:
    camera_data = json.load(f)
    serials = camera_data["serials"]
    extrinsics = camera_data.get("extrinsics", {})

# Initialize RealSense pipeline for cameras
pipelines, aligns, profiles = initialize_cameras(serials=serials, resolution=[640, 360])

# Initialize a YOLO model for each camera
models = [YOLO("yolov9e-seg.pt") for _ in range(len(pipelines))]

# Store annotated frames and 3D coordinates for each camera
annotated_frames = [None] * len(pipelines)
robot_coordinates_list = [None] * len(pipelines)

def process_camera(pipeline, align, profile, model, idx, camera_transform):
    global annotated_frames, robot_coordinates_list
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

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
        else:
            annotated_frame = color_image.copy()  # Ensure annotated_frame is always defined

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

                        # Convert depth and pixel coordinates to 3D coordinates
                        x = (center_x - intrinsics.ppx) * average_depth / intrinsics.fx
                        y = (center_y - intrinsics.ppy) * average_depth / intrinsics.fy
                        z = average_depth

                        # Transform from camera coordinates to robot/world coordinates using extrinsics
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
        
        if stop_threads:
            break

# Create and start a thread for each camera
threads = []
stop_threads = False
for i, (pipeline, align, profile, model, serial) in enumerate(zip(pipelines, aligns, profiles, models, serials)):
    camera_transform = np.array(extrinsics.get(serial, np.eye(4)))  # Default to identity matrix if not provided
    thread = threading.Thread(target=process_camera, args=(pipeline, align, profile, model, i, camera_transform))
    threads.append(thread)
    thread.start()

# Initialize matplotlib for 3D plotting outside of the function
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 5])
    ax.set_zlim([-1, 1])

    # Create scatter plot and text objects that can be updated
    scatter = ax.scatter([], [], [], c='r', marker='o')

# Display the combined results
try:
    while True:
        # Combine the annotated frames side by side (when available)
        if all(frame is not None for frame in annotated_frames) and draw:
            combined_frame = np.hstack(annotated_frames)
            cv2.imshow("YOLO Detection with Depth and Masks", combined_frame)

        # Update the 3D plot with robot coordinates from each camera
        if plot and all(coords is not None for coords in robot_coordinates_list):
            all_x_robot, all_y_robot, all_z_robot, all_object_ids = [], [], [], []
            for coords in robot_coordinates_list:
                x_robot, y_robot, z_robot, object_ids = coords
                all_x_robot.extend(x_robot)
                all_y_robot.extend(y_robot)
                all_z_robot.extend(z_robot)
                all_object_ids.extend(object_ids)
            update_plot(scatter, ax, all_x_robot, all_y_robot, all_z_robot, [])

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Stop threads and release resources
    stop_threads = True
    for thread in threads:
        thread.join()
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
