import time
import numpy as np
from MultiDetector import MultiDetector  # Assuming your class is saved in Python_Realsense_Multi.py
import threading
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Merger:
    def __init__(self, camera_config_path, detection_type='mask', threshold=0.1, show=True, plot=True):
        self.camera_config_path = camera_config_path
        self.detection_type = detection_type
        self.threshold = threshold
        self.show = show
        self.plot = plot
        self.detector = MultiDetector(camera_config_path=self.camera_config_path, detection_type=self.detection_type, show=True, draw=True, plot=False, verbose=False)
        if plot:
            self._init_plot()

    def _init_plot(self):
        """Initialize 2D plotting with matplotlib."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))  # 2D plot
        self.ax.set_xlabel('X')  # X-axis
        self.ax.set_ylabel('Z')  # Z on the Y-axis

        # Set axis limits
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])  # Z-axis limits now on Y-axis

        # Add an arrow in the positive Z direction at (0, 0)
        self.ax.annotate('', xy=(0, 0), xytext=(0, -0.3),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10))

        # Initialize scatter objects for single and merged detections
        self.single_scatter = self.ax.scatter([], [], c='b', marker='o', label='Single')
        self.merged_scatter = self.ax.scatter([], [], c='r', marker='o', label='Merged')

        # Create a custom arrow legend entry using Line2D
        arrow_legend = Line2D([0], [0], color='green', marker='|', linestyle='None', markersize=10, label='Robot')

        # Add the arrow to the legend along with scatter plots
        self.ax.legend(handles=[self.single_scatter, self.merged_scatter, arrow_legend], loc='upper right')


    def calculate_average_position(self, detections):
        """Calculate the average position of detections."""
        avg_x = np.mean([d[0] for d in detections])
        avg_y = np.mean([d[1] for d in detections])
        avg_z = np.mean([d[2] for d in detections])
        return [avg_x, avg_y, avg_z]

    def find_similar_detections(self, detection_list):
        """Group detections by object names and then compare positions across different cameras."""
        name_groups = {}
        
        for detection in detection_list:
            obj_x, obj_y, obj_z, object_ids, object_names, cam_idx = detection
            
            for name, obj_id, x, y, z in zip(object_names, object_ids, obj_x, obj_y, obj_z):
                if name not in name_groups:
                    name_groups[name] = {
                        'cam_idx': [cam_idx],
                        'object_id': [obj_id],
                        'Position': [[x, y, z]],
                    }
                else:
                    name_groups[name]['cam_idx'].append(cam_idx)
                    name_groups[name]['object_id'].append(obj_id)
                    name_groups[name]['Position'].append([x, y, z])

        similar_detections = []
        for name, data in name_groups.items():
            positions = data['Position']
            cam_indices = data['cam_idx']
            object_ids = data['object_id']
            group = []
            seen = set()
            
            for i in range(len(object_ids)):
                if object_ids[i] in seen:
                    continue
                
                current_position = np.array(positions[i])
                current_cam_idx = cam_indices[i]
                current_obj_id = object_ids[i]
                
                similar_group = {
                    'object_name': name,
                    'object_id': [current_obj_id],
                    'positions': [current_position],
                    'cameras': [current_cam_idx]
                }
                seen.add(object_ids[i])

                for j in range(i + 1, len(positions)):
                    if object_ids[j] in seen:
                        continue
                    
                    other_position = np.array(positions[j])
                    other_cam_idx = cam_indices[j]
                    other_obj_id = object_ids[j]

                    distance = np.linalg.norm(current_position - other_position)

                    if distance < self.threshold and current_cam_idx != other_cam_idx:
                        similar_group['object_id'].append(other_obj_id)
                        similar_group['positions'].append(other_position)
                        similar_group['cameras'].append(other_cam_idx)
                        seen.add(object_ids[j])

                if len(similar_group['positions']) > 1:
                    avg_position = self.calculate_average_position(similar_group['positions'])
                    similar_group['average_position'] = avg_position

                group.append(similar_group)
            
            similar_detections.extend(group)
        
        return similar_detections

    def plot_2d_results(self, similar_detections):
        """Plot the raw detections and the merged results in a 2D plot."""
        
        # Separate x, z coordinates for raw and merged positions (since Z is now on the Y-axis)
        pos_x, pos_z = [], []
        avg_x, avg_z = [], []

        # Organize detection data into the correct format for plotting
        for det in similar_detections:
            if len(det['object_id']) > 1:  # Merged detections
                avg_pos = det['average_position']
                avg_x.append(float(avg_pos[0]))  # X
                avg_z.append(float(avg_pos[2]))  # Z (on Y-axis)
            else:  # Single detections
                pos = det['positions'][0]
                pos_x.append(pos[0])  # X
                pos_z.append(pos[2])  # Z (on Y-axis)
            
        # Update scatter plot data
        self.single_scatter.set_offsets(np.c_[pos_x, pos_z])
        self.merged_scatter.set_offsets(np.c_[avg_x, avg_z])

        # Redraw the plot
        plt.draw()
        plt.pause(0.001)  # This is required to keep the plot interactive

    def merge_and_plot_detections(self):
        """Merge detections and plot 3D results."""
        self.detector.start()

        try:
            while True:
                combined_detections = []
                for idx, detection in enumerate(self.detector.detection_list):
                    if detection is not None:
                        combined_detections.append(detection)

                similar_detections = self.find_similar_detections(combined_detections)

                # Print merged detections with similar names and positions within the threshold
                print("\n--- Merged Detections (within threshold) ---")
                if not similar_detections:
                    print("No merged detections found within the threshold.")
                for det in similar_detections:
                    if len(det['object_id']) > 1:
                        avg_pos = det['average_position']
                        print(f"Detected duplicate {det['object_name']} (ID: {det['object_id']}) at average position {avg_pos} from cameras {det['cameras']}")
                    else:
                        pos = det['positions'][0]
                        print(f"Detected {det['object_name']} (ID: {det['object_id']}) at position {pos} from camera {det['cameras']}")

                if self. plot: self.plot_2d_results(similar_detections)
                if self.show: self.detector.show_combined_frames()
                time.sleep(0.1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            plt.close()
            self.detector.stop()

if __name__ == "__main__":
    merger = Merger(camera_config_path='CAMERAS_Jackal04.json', threshold=0.35, show=True, plot=True)
    merger.merge_and_plot_detections()
