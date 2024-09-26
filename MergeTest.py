import time
import numpy as np
from Python_Realsense_Multi import MultiDetection  # Assuming your class is saved in Python_Realsense_Multi.py
import threading
import cv2

def calculate_average_position(detections):
    """Calculate the average position of detections."""
    avg_x = "%.5f" % np.mean([d[0] for d in detections])
    avg_y = "%.5f" % np.mean([d[1] for d in detections])
    avg_z = "%.5f" % np.mean([d[2] for d in detections])
    return avg_x, avg_y, avg_z

def find_similar_detections(detection_list, threshold=0.1):
    """Group detections by object names and then compare positions across different cameras."""
    
    # Step 1: Group detections by object names
    name_groups = {}
    
    for detection in detection_list:
        obj_x, obj_y, obj_z, object_ids, object_names, cam_idx = detection
        
        for name, obj_id, x, y, z in zip(object_names, object_ids, obj_x, obj_y, obj_z):
            # Initialize the group if it doesn't exist
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

    # Step 2: Compare detections within each group
    similar_detections = []

    for name, data in name_groups.items():
        positions = data['Position']
        cam_indices = data['cam_idx']
        object_ids = data['object_id']
        
        # Group similar detections by checking the distance between positions
        group = []
        seen = set()  # To track which detections have already been processed
        
        for i in range(len(object_ids)):
            if object_ids[i] in seen:
                continue  # Skip already processed positions
            
            current_position = np.array(positions[i])
            current_cam_idx = cam_indices[i]
            current_obj_id = object_ids[i]
            
            # Start a new group with the current detection
            similar_group = {
                'object_name': name,
                'object_id': [current_obj_id],
                'positions': [current_position],
                'cameras': [current_cam_idx]
            }
            seen.add(object_ids[i])

            # Compare current detection with all others in the same group
            for j in range(i + 1, len(positions)):
                if object_ids[j] in seen:
                    continue
                
                other_position = np.array(positions[j])
                other_cam_idx = cam_indices[j]
                other_obj_id = object_ids[j]

                # Calculate the distance between the current detection and the other detection
                distance = np.linalg.norm(current_position - other_position)

                # If the distance is within the threshold, group the detections together
                if distance < threshold and current_cam_idx != other_cam_idx:
                    similar_group['object_id'].append(other_obj_id)
                    similar_group['positions'].append(other_position)
                    similar_group['cameras'].append(other_cam_idx)
                    seen.add(object_ids[j])
            
            # Calculate average position if multiple similar detections are found
            if len(similar_group['positions']) > 1:
                avg_position = calculate_average_position(similar_group['positions'])
                similar_group['average_position'] = avg_position

            group.append(similar_group)
        
        similar_detections.extend(group)
    
    return similar_detections

def merge_detection(threshold):
    # Load camera configuration from JSON file
    camera_config_path = 'CAMERAS_doubletest.json'

    # Initialize the MultiDetection classq
    detector = MultiDetection(camera_config_path=camera_config_path, detection_type='mask', show=True, draw=True, plot=True, verbose=False)

    try:
        # Start the detection process
        detector.start()

        # Collect the detection results
        while True:
            combined_detections = []
            for idx, detection in enumerate(detector.detection_list):
                if detection is not None:
                    combined_detections.append(detection)

            # Find similar detections across cameras
            similar_detections = find_similar_detections(combined_detections, threshold)  # Adjust threshold as needed

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

            time.sleep(0.1)
            
            detector.show_combined_frames()
            detector.update_3d_plot()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Stop the detector to clean up
        detector.stop()

if __name__ == "__main__":
    merge_detection(threshold=0.3)
