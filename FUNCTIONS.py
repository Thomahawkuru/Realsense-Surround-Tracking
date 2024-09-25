from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs

def compute_trapezoid_points(image_shape, tilt_angle):
    height, width = image_shape[:2]
    angle_rad = np.deg2rad(-tilt_angle)
    
    # Calculate the height offset at the top of the image due to tilt
    offset = height * np.tan(angle_rad)
    
    # Define the trapezoid points in the distorted image
    src_points = np.array([
        [offset, height],          # Bottom-left
        [width - offset, height],  # Bottom-right
        [width, 0],                # Top-right
        [0, 0]                     # Top-left
    ], dtype=np.float32)
    
    # Define the corresponding rectangle points in the rectified image
    dst_points = np.array([
        [0, height],               # Bottom-left
        [width, height],           # Bottom-right
        [width, 0],                # Top-right
        [0, 0]                     # Top-left
    ], dtype=np.float32)
    
    return src_points, dst_points

def rectify_image(image, tilt_angle):
    # Compute trapezoid points based on the tilt angle
    src_points, dst_points = compute_trapezoid_points(image.shape, tilt_angle)
    
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective warp
    rectified_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    return rectified_image

# Function to remove outliers based on IQR
def remove_outliers(data):
    if len(data) == 0:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def update_plot(scatter, ax, x, z, y, ids):   
    # Update scatter plot
    scatter._offsets3d = (x, y, z)
    for i, txt in enumerate(ids):
        ax.text(x[i], y[i], z[i], txt, fontsize=8, color='blue')
    
    plt.draw()
    plt.pause(0.001)

def crop_image(image, crop_pixels):
    return image[:, crop_pixels:-crop_pixels]

def initialize_cameras(serials, resolution=(640*2, 360*2), fps=30):
    """
    Initializes RealSense pipelines for multiple cameras and returns the pipelines, align objects, and camera parameters.
    
    Parameters:
        serials (list of str): List of serial numbers for the RealSense cameras.
        resolution (tuple): Resolution of the depth stream.
        fps (int): Frames per second.

    Returns:
        tuple: (pipelines, aligns, Profiles)
    """
    print(serials)
    pipelines = []
    aligns = []
    profiles = []
    
    for serial in serials:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, *resolution, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, *resolution, rs.format.bgr8, fps)
        
        profiles.append(pipeline.start(config))
        align = rs.align(rs.stream.color)
        
        pipelines.append(pipeline)
        aligns.append(align)
    
    return pipelines, aligns, profiles

def run_tracker_in_thread(model_name, image):
    """
    Run YOLO tracker in its own thread for concurrent processing.

    Args:
        model_name (str): The YOLOv8 model object.
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
    """
    model = YOLO(model_name)
    results = model.track(image, save=True, stream=True)
    for r in results:
        pass