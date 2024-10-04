import cv2
import numpy as np
import os
from playsound import playsound

def draw_bboxes(frame, bboxes, is_fall):
    color = (0, 0, 255) if is_fall else (0, 255, 0)  # Red for fall, Green otherwise
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "FALL DETECTED" if is_fall else "Human"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def play_alert_sound():
    # Path to sound file
    sound_path = 'alert_sound.mp3'
    
    # Play the sound (make sure the sound file exists)
    if os.path.exists(sound_path):
        playsound(sound_path)
