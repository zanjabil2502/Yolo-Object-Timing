import cv2
from ultralytics import YOLO
from src.object_timer import ObjectTimer

path_model = "yolov8n.pt"
url = "http://103.151.177.238:1004/axis-cgi/media.cgi?audiocodec=aac&audiosamplerate=16000&audiobitrate=32000&camera=1&videoframeskipmode=empty&videozprofile=classic&resolution=1280x960"

# Initialize YOLOv8 Detection Model
model = YOLO(path_model)
classes_to_count = [2]  # car classes for count
region_points = [(0, 600), (350, 400), (1280, 400), (1280, 960), (0, 960)]

# Open the Video File
cap = cv2.VideoCapture(url)
assert cap.isOpened(), "Error reading video file"
fps = cap.get(cv2.CAP_PROP_FPS)


# Initialize Object Counter
timer = ObjectTimer(
    fps=fps,  # FPS from you source video
    view_img=True,  # Display the image during processing
    reg_pts=region_points,  # Region of interest points
    names=model.names,  # Class names from the YOLO model
    draw_tracks=True,  # Draw tracking lines for objects
    line_thickness=2,  # Thickness of the lines drawn
)


# itterate over video frames:
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    # Write the annotated frame to the output video
    frame_count += 1

    # Perform object tracking on the current frame
    tracks = model.track(frame, persist=True, classes=classes_to_count)

    # Use the Object Counter to count objects in the frame and get the annotated image
    frame = timer.start_timing(frame, tracks, frame_count)

# Release all Resources:
cap.release()
cv2.destroyAllWindows()
