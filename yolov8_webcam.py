import cv2
from ultralytics import YOLO

# Load a YOLOv8 model (using the small version for speed)
model = YOLO('yolov8n.pt')  # You can change to 'yolov8s.pt', 'yolov8m.pt', etc.

# Open the default webcam (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 