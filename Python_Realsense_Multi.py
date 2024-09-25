import cv2
import numpy as np
import json
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
import matplotlib.pyplot as plt

class MultiDetection:
    def __init__(self, camera_config_path='CAMERAS.json', detection_type='box', show=True, draw=False, plot=False, verbose=False):
        self.show = show
        self.draw = draw
        self.plot = plot
        self.verbose = verbose
        self.detection_type = detection_type
        self.stop_threads = False

        # Load camera serials and extrinsics
        with open(camera_config_path, 'r') as f:
            camera_data = json.load(f)
        self.serials = camera_data["serials"]
        self.extrinsics = camera_data.get("extrinsics", {})

        # Initialize RealSense pipeline
        self.pipelines, self.aligns, self.profiles = self._initialize_cameras(resolution=[640, 360], fps=15)

        # Initialize YOLO models
        self.models = self._initialize_yolo_models()

        # Annotated frames and detection list
        self.annotated_frames = [None] * len(self.pipelines)
        self.detection_list = [None] * len(self.pipelines)

        # Initialize 3D plotting, if enabled
        if self.plot:
            self._init_3d_plot()

    def _initialize_cameras(self, resolution=(640*2, 360*2), fps=30):
        """ Initializes RealSense camera pipelines and aligns them for multi-camera setups. """
        pipelines = []
        aligns = []
        profiles = []
        print(self.serials)
        for serial in self.serials:
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

    def _initialize_yolo_models(self):
        """Initialize YOLO models based on the detection type."""
        if self.detection_type == 'box':
            return [YOLO("yolov10x.pt", verbose=self.verbose) for _ in range(len(self.pipelines))]
        elif self.detection_type == 'mask':
            return [YOLO("yolov9e-seg.pt", verbose=self.verbose) for _ in range(len(self.pipelines))]

    def _init_3d_plot(self):
        """Initialize 3D plotting with matplotlib."""
        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-1, 2])

        self.scatters = []
        colors = ['r', 'b', 'g', 'y', 'k']
        labels = [f'Camera {i + 1}' for i in range(len(self.serials))]

        for i in range(len(self.serials)):
            self.scatters.append(self.ax.scatter([], [], [], c=colors[i], marker='o', label=labels[i]))

        self.ax.legend(loc='upper right')

    def process_camera(self, pipeline, align, profile, model, idx, camera_transform):
        """Camera processing function for threading."""
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        while not self.stop_threads:
            # Get frames
            frame = pipeline.wait_for_frames()
            aligned_frame = align.process(frame)
            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Run YOLO tracking
            results = model.track(color_image, persist=True, verbose=self.verbose)

            # Visualization
            if self.draw:
                if self.detection_type == 'box':
                    annotated_frame = results[0].plot(boxes=True, masks=False)
                elif self.detection_type == 'mask':
                    annotated_frame = results[0].plot(boxes=False, masks=True)
            else:
                annotated_frame = color_image.copy()

            # Process results based on detection type
            if self.detection_type == 'box':
                x_robot, y_robot, z_robot, object_ids, object_names = self._process_results_box(results, depth_image, depth_frame, annotated_frame, intrinsics, camera_transform)
            elif self.detection_type == 'mask':
                x_robot, y_robot, z_robot, object_ids, object_names = self._process_results_mask(results, depth_image, depth_frame, annotated_frame, intrinsics, camera_transform)

            self.annotated_frames[idx] = annotated_frame
            self.detection_list[idx] = (x_robot, z_robot, y_robot, object_ids, object_names, idx + 1)

    def _process_results_box(self, results, depth_image, depth_frame, annotated_frame, intrinsics, camera_transform):
        """Process YOLO results and compute 3D coordinates."""
        x_robot, y_robot, z_robot, object_ids, object_names = [], [], [], [], []

        for result in results:
            if result.boxes:
                boxes = result.boxes
                classes = result.boxes.cls
                names = result.names
                ids = boxes.id.tolist() if boxes.id is not None else []

                for i, (box, obj_id, obj_class) in enumerate(zip(boxes.xyxy, ids, classes)):
                    obj_name = names[obj_class.item()]
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2
                    depth_at_center = depth_image[int(center_y), int(center_x)] * depth_frame.get_units()

                    x, y, z = self._calculate_3d_coordinates(center_x, center_y, depth_at_center, intrinsics)
                    robot_coordinates = np.dot(camera_transform, np.array([x, y, z, 1]))
                    x_robot.append(robot_coordinates[0])
                    y_robot.append(robot_coordinates[1])
                    z_robot.append(robot_coordinates[2])
                    object_ids.append(f"ID {obj_id}")
                    object_names.append(obj_name)

                    if self.draw and self.show:
                        # Annotate the frame with the object's robot position
                        label = f"ID {obj_id} ({obj_name}):\n({x:.2f}, {y:.2f}, {z:.2f})"
                        lines = label.split('\n')
                        for j, line in enumerate(lines):
                            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            text_width, text_height = text_size
                            text_x = int(center_x - text_width // 2)
                            text_y = int(center_y + (j * 3 + 1) * text_height // 2)
                            cv2.putText(annotated_frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            return x_robot, y_robot, z_robot, object_ids, object_names
    
    def _process_results_mask(self, results, depth_image, depth_frame, annotated_frame, intrinsics, camera_transform):
        """Process YOLO mask results and compute 3D coordinates."""
        x_robot, y_robot, z_robot, object_ids, object_names = [], [], [], [], []

        for result in results:
            if result.masks:    
                masks = result.masks.data.cpu().numpy() 
                classes = result.boxes.cls
                names = result.names
                ids = result.boxes.id.tolist() if result.boxes.id is not None else []

                for i, (mask, obj_id, obj_class) in enumerate(zip(masks, ids, classes)):
                    obj_name = names[obj_class.item()]
                    mask_indices = np.where(mask == 1)
                    mask_indices = (np.clip(mask_indices[0], 0, depth_image.shape[0] - 1),
                                    np.clip(mask_indices[1], 0, depth_image.shape[1] - 1))
                    object_depth_values = depth_image[mask_indices] * depth_frame.get_units()
                    average_depth = np.mean(object_depth_values) if object_depth_values.size > 0 else 0
                    if mask_indices[0].size > 0 and mask_indices[1].size > 0:
                        center_x = int(np.mean(mask_indices[1]))  # X coordinate (columns)
                        center_y = int(np.mean(mask_indices[0]))  # Y coordinate (rows)
                    else:
                        center_x, center_y = 0, 0
                    
                    x, y, z = self._calculate_3d_coordinates(center_x, center_y, average_depth, intrinsics)
                    robot_coordinates = np.dot(camera_transform, np.array([x, y, z, 1]))
                    x_robot.append(robot_coordinates[0])
                    y_robot.append(robot_coordinates[1])
                    z_robot.append(robot_coordinates[2])
                    object_ids.append(f"ID {obj_id}")
                    object_names.append(obj_name)
                    
                    if self.draw and self.show:
                        # Annotate the frame with the object's robot position
                        label = f"ID {obj_id} ({obj_name}):\n({x:.2f}, {y:.2f}, {z:.2f})"
                        lines = label.split('\n')
                        for j, line in enumerate(lines):
                            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            text_width, text_height = text_size
                            text_x = int(center_x - text_width // 2)
                            text_y = int(center_y + (j * 3 + 1) * text_height // 2)
                            cv2.putText(annotated_frame, line, (text_x, text_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return x_robot, y_robot, z_robot, object_ids, object_names
    
    def _calculate_3d_coordinates(self, center_x, center_y, depth, intrinsics):
        """Convert pixel coordinates and depth to 3D coordinates."""
        x = (center_x - intrinsics.ppx) * depth / intrinsics.fx
        y = (center_y - intrinsics.ppy) * depth / intrinsics.fy
        z = depth
        return x, y, z
    
    def start(self):
        """Start YOLO detection with threading for each camera."""
        self.threads = []
        for i, (pipeline, align, profile, model, serial) in enumerate(zip(self.pipelines, self.aligns, self.profiles, self.models, self.serials)):
            camera_transform = np.array(self.extrinsics.get(serial, np.eye(4)))
            thread = threading.Thread(target=self.process_camera, args=(pipeline, align, profile, model, i, camera_transform))
            thread.start()
            self.threads.append(thread)
    
    def show_results(self):
        """Show and or Draw the detection results in plots and images."""
        try:
            while True:
                self._show_combined_frames()
                self._update_3d_plot()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stop()
    
    def _show_combined_frames(self):
        """Combine and display annotated frames."""
        if self.show and all(frame is not None for frame in self.annotated_frames):
            try:
                reordered_frames = self.annotated_frames[3:5] + self.annotated_frames[0:3]
            except IndexError:  # Only catch IndexErrors
                reordered_frames = self.annotated_frames
            combined_frame = np.hstack(reordered_frames)
            cv2.imshow("YOLO Detection with Depth and Masks", combined_frame)

    def _update_3d_plot(self):
        """Update the 3D plot with detection data."""
        if self.plot and all(coords is not None for coords in self.detection_list):
            for scatter, detection in zip(self.scatters, self.detection_list):
                scatter._offsets3d = detection[:3]
                plt.draw()
                plt.pause(0.001)

    def stop(self):
        """Stop the threads and clean up."""
        self.stop_threads = True
        for thread in self.threads:
            thread.join()
        for pipeline in self.pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    try:
        detector = MultiDetection(camera_config_path='CAMERAS.json', detection_type='mask', show=True, draw=True, plot=False, verbose=True)
        detector.start()
        detector.show_results()
    except:
        print('Exception occured')