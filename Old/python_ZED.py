from collections import defaultdict
import cv2
from ultralytics import YOLO, SAM, RTDETR
from ultralytics.utils.plotting import Annotator, colors

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the model with segmentation capabilities
# model = SAM("sam2_b.pt")
model = YOLO("yolov9e-seg.pt")
# model = YOLO("yolov10x.pt")
# model = RTDETR("rtdetr-l.pt")

# Open the webcam feed (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Set webcam resolution to 720p (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920*2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Retrieve webcam properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video with the specified properties
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Get the left quarter (first 960 pixels horizontally)
        left_quarter = frame[:, :960]

        # Get the right quarter (last 960 pixels horizontally)
        right_quarter = frame[:, -960:]

        # Concatenate the left and right quarters horizontally to form a 1920x1080 frame
        glued_frame = cv2.hconcat([left_quarter, right_quarter])
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(glued_frame,persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes=True, masks=True)

        # Display the annotated frame
        cv2.imshow("YOLOv10 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()