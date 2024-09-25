from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs

def compute_trapezoid_points(image_shape, tilt_angle):
    """
    Computes the trapezoidal points in an image based on its shape and a tilt angle.

    Args:
        image_shape (tuple): The dimensions of the image (height, width, channels).
        tilt_angle (float): The angle in degrees to simulate the tilt of the camera.

    Returns:
        src_points (np.array): The trapezoid points in the original (distorted) image.
        dst_points (np.array): The corresponding rectangular points in the corrected image.
    """
    height, width = image_shape[:2]
    angle_rad = np.deg2rad(-tilt_angle)
    
    # Calculate the vertical offset due to the tilt angle
    offset = height * np.tan(angle_rad)
    
    # Define the trapezoidal points based on the tilt (bottom becomes wider)
    src_points = np.array([
        [offset, height],          # Bottom-left
        [width - offset, height],  # Bottom-right
        [width, 0],                # Top-right
        [0, 0]                     # Top-left
    ], dtype=np.float32)
    
    # Define corresponding rectangular points in the rectified image
    dst_points = np.array([
        [0, height],               # Bottom-left
        [width, height],           # Bottom-right
        [width, 0],                # Top-right
        [0, 0]                     # Top-left
    ], dtype=np.float32)
    
    return src_points, dst_points

def rectify_image(image, tilt_angle):
    """
    Rectifies an image by correcting its perspective based on the camera's tilt angle.

    Args:
        image (np.array): The input image to be rectified.
        tilt_angle (float): The tilt angle of the camera.

    Returns:
        rectified_image (np.array): The rectified image with a straightened perspective.
    """
    # Compute trapezoid points based on image tilt
    src_points, dst_points = compute_trapezoid_points(image.shape, tilt_angle)
    
    # Compute the perspective transformation matrix from source to destination points
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective warp to the input image
    rectified_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    return rectified_image

def remove_outliers(data):
    """
    Removes outliers from a dataset using the Interquartile Range (IQR) method.

    Args:
        data (np.array): The dataset from which outliers are to be removed.

    Returns:
        np.array: The dataset with outliers removed.
    """
    if len(data) == 0:
        return data
    
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def update_plot(scatter, ax, x, z, y, ids):   
    """
    Updates the 3D scatter plot with new positions and labels.

    Args:
        scatter (matplotlib Axes3D.scatter): The scatter plot object to update.
        ax (matplotlib Axes3D): The axes object for the plot.
        x (list): X coordinates of the objects.
        z (list): Z coordinates of the objects.
        y (list): Y coordinates of the objects.
        ids (list): List of object IDs to annotate the plot.
    """
    # Update scatter plot coordinates
    scatter._offsets3d = (x, y, z)
    
    # Add text labels next to the points in the 3D plot
    for i, txt in enumerate(ids):
        ax.text(x[i], y[i], z[i], txt, fontsize=8, color='blue')
    
    # Redraw and pause to allow updates in real-time
    plt.draw()
    plt.pause(0.001)

def crop_image(image, crop_pixels):
    """
    Crops an image by removing a specified number of pixels from both left and right sides.

    Args:
        image (np.array): The input image to crop.
        crop_pixels (int): The number of pixels to remove from both sides.

    Returns:
        np.array: The cropped image.
    """
    return image[:, crop_pixels:-crop_pixels]

def initialize_cameras(serials, resolution=(640*2, 360*2), fps=30):
    """
    Initializes RealSense camera pipelines and aligns them for multi-camera setups.

    Args:
        serials (list of str): List of RealSense camera serial numbers to initialize.
        resolution (tuple): The resolution of the camera stream (width, height).
        fps (int): The frame rate for the camera stream.

    Returns:
        tuple: 
            - pipelines (list): A list of initialized camera pipelines.
            - aligns (list): A list of align objects for depth and color alignment.
            - profiles (list): A list of camera profiles, including calibration info.
    """
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
    Runs the YOLO model in a separate thread for real-time object tracking.

    Args:
        model_name (str): The name of the YOLO model to use.
        image (np.array): The input image on which tracking is performed.
    """
    model = YOLO(model_name)
    results = model.track(image, save=True, stream=True)
    
    for r in results:
        pass  # Process or display results as needed
