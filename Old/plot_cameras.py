import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def plot_camera_extrinsics_from_file(filename):
    # Read data from the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    serials = data["serials"]
    extrinsics = data["extrinsics"]
    
    # Prepare figure
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different cameras
    colors = {
        serials[0]: 'r',
        serials[1]: 'g',
        serials[2]: 'b',
        serials[3]: 'y',
        serials[4]: 'm'
    }
    
    # Prepare to store camera positions
    positions = []
    
    # Plot each camera
    for i, (serial) in enumerate(serials):
        if serial in extrinsics:
            extrinsic_matrix = np.array(extrinsics[serial])
            # Extract position from the last column of the extrinsic matrix
            position = extrinsic_matrix[:3, 3]
            positions.append(position)  # Store position for equal axis setting
            
            # Plot camera position
            ax.scatter(position[0], position[1], position[2], color=colors.get(serial, 'k'), label=f'Camera {i+1}, {serial}')
            # Draw camera direction (optional, assume Z is forward direction)
            direction = extrinsic_matrix[:3, 2]  # Forward direction
            ax.quiver(position[0], position[1], position[2], 
                       direction[0], direction[1], direction[2], 
                       length=0.1, color=colors.get(serial, 'k'))

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Camera Extrinsics')
    ax.legend()

    # Set axes equal
    max_range = 0.2
    mid_x = (np.max(positions, axis=0)[0] + np.min(positions, axis=0)[0]) * 0.5
    mid_y = (np.max(positions, axis=0)[1] + np.min(positions, axis=0)[1]) * 0.5
    mid_z = (np.max(positions, axis=0)[2] + np.min(positions, axis=0)[2]) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

# Example usage
plot_camera_extrinsics_from_file('CAMERAS_Jackal04.json')
