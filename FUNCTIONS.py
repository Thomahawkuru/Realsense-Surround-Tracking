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

def update_plot(x, y, z, ids):
    # Initialize matplotlib for 3D plotting
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Clear previous plot
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 5])
    ax.set_zlim([-1, 1])
    
    # Update scatter plot
    scatter = ax.scatter(x, y, z, c='r', marker='o')
    for i, txt in enumerate(ids):
        ax.text(x[i], y[i], z[i], txt, fontsize=8, color='blue')
    
    plt.draw()
    plt.pause(0.001)

def crop_image(image, crop_pixels):
    return image[:, crop_pixels:-crop_pixels]

def initialize_cameras(serials, depth_resolution=(640*2, 360*2), color_resolution=(640*2, 360*2), fps=30):
    """
    Initializes RealSense pipelines for multiple cameras and returns the pipelines, align objects, and camera parameters.
    
    Parameters:
        serials (list of str): List of serial numbers for the RealSense cameras.
        depth_resolution (tuple): Resolution of the depth stream.
        color_resolution (tuple): Resolution of the color stream.
        fps (int): Frames per second.

    Returns:
        tuple: (pipelines, aligns, fx, fy, cx, cy, camera_to_robot_transform)
    """
    pipelines = []
    aligns = []
    
    for serial in serials:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, *depth_resolution, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, *color_resolution, rs.format.bgr8, fps)
        
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        
        pipelines.append(pipeline)
        aligns.append(align)
    
    return pipelines, aligns