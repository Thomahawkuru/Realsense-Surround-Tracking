from collections import defaultdict
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.ndimage import center_of_mass
from ultralytics.utils.plotting import colors

# Initialize YOLOv9 segmentation model
model = YOLO("yolov9e-seg.pt")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Create align object to align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Open the webcam feed (using the RealSense camera here)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define strip width
strip_width = 25

# Function to remove outliers based on IQR
def remove_outliers(data):
    if len(data) == 0:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Processing loop
try:
    while cap.isOpened():
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

        # Define center and strip coordinates
        height, width = color_image.shape[:2]
        start_x = width // 2 - strip_width // 2
        end_x = width // 2 + strip_width // 2

        # Remove the center strip from both color and depth images
        color_image_left = color_image[:, :start_x]
        color_image_right = color_image[:, end_x:]
        color_image_stitched = np.hstack((color_image_left, color_image_right))

        depth_image_left = depth_image[:, :start_x]
        depth_image_right = depth_image[:, end_x:]
        depth_image_stitched = np.hstack((depth_image_left, depth_image_right))

        # Run YOLOv9 tracking on the stitched color image
        results = model.track(color_image_stitched, persist=True)

        # Visualize the results on the frame (draw bounding boxes and masks)
        annotated_frame = results[0].plot(boxes=True, masks=True)

        # Create a blank mask for depth image (full resolution)
        depth_mask = np.zeros((depth_image_stitched.shape[0], depth_image_stitched.shape[1]), dtype=np.uint8)

        # Loop through each detected object
        for result in results:
            # Check if segmentation masks are available
            if result.masks and result.boxes:
                # Loop over all detected objects in the current frame
                for i, mask in enumerate(result.masks.data.cpu().numpy()):
                    # Upscale the mask from half resolution to full resolution
                    mask_upscaled = cv2.resize(mask, (depth_image_stitched.shape[1], depth_image_stitched.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Get the mask indices where the mask is active (i.e., the object area)
                    mask_indices = np.where(mask_upscaled == 1)

                    # Fill the depth_mask with the upscaled mask
                    depth_mask[mask_indices] = 1

                    # Extract depth values for the object using the upscaled mask
                    object_depth_values = remove_outliers(depth_image_stitched[mask_indices])

                    # Calculate the average depth of the object
                    if object_depth_values.size > 0:
                        average_depth = np.mean(object_depth_values) * depth_frame.get_units()
                    else:
                        average_depth = 0

                    # Calculate the centroid of the mask
                    if mask_upscaled.size > 0:
                        # Calculate center of mass for the mask
                        center_mass = center_of_mass(mask_upscaled)
                        center_x, center_y = int(center_mass[1]), int(center_mass[0])
                    else:
                        center_x, center_y = 0, 0

                    # Annotate the frame with the object's depth at the center of the mask
                    label = f"Depth: {average_depth:.2f} meters"
                    
                    # Calculate text size
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_width, text_height = text_size
                    
                    # Calculate the text position (centered)
                    text_x = max(center_x - text_width // 2, 0)  # Ensure text doesn't go out of the image boundary
                    text_y = max(center_y + text_height // 2, 0)  # Ensure text doesn't go out of the image boundary

                    # Use cv2.putText to add depth information at the center of the mask
                    cv2.putText(annotated_frame, label, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the annotated frame with bounding boxes and masks
        cv2.imshow("YOLOv9 Detection with Depth and Masks", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    pipeline.stop()
    cap.release()
    cv2.destroyAllWindows()
