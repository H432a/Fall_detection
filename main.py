import torch
import cv2
import numpy as np
from fall_detection import detect_fall
from utils import draw_bboxes, play_alert_sound

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8s.pt')

def main():
    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(0)  # 0 for webcam, replace with file path for video

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        results = model(frame)
        bboxes = results.xyxy[0].numpy()  # Bounding boxes

        # Detect human (class 0)
        humans = [bbox for bbox in bboxes if int(bbox[5]) == 0]
        
        # ML-based fall detection
        is_fall = detect_fall(humans)

        # Draw bounding boxes and alert
        frame = draw_bboxes(frame, humans, is_fall)

        # Play sound if fall is detected
        if is_fall:
            play_alert_sound()

        # Show frame with alerts
        cv2.imshow('Real-Time Fall Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
