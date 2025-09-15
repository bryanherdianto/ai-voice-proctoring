import cv2
import os

os.makedirs("assets/frames", exist_ok=True)

cap = cv2.VideoCapture("assets/video_sample.mp4")
fps = 0.1
frame_rate = cap.get(cv2.CAP_PROP_FPS)
interval = int(frame_rate // fps)
count = 0
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % interval == 0:
        frame_filename = f"assets/frames/frame_{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
        frame_number += 1
    count += 1

cap.release()
print(f"Extracted {frame_number} frames")