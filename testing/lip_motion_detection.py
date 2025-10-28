import cv2
import mediapipe as mp
import numpy as np
import glob

mp_face = mp.solutions.face_mesh
lips_motion = []
previous_mouth = None

frame_files = sorted(glob.glob("assets/frames/*.jpg"))

if not frame_files:
    print("No frame files found in assets/frames/")
    exit()

print(f"Processing {len(frame_files)} frames...")

with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Could not read frame: {frame_path}")
            lips_motion.append(0)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            # Get lip landmarks (inner and outer lips)
            mouth = np.array(
                [
                    (lm.x, lm.y)
                    for i, lm in enumerate(landmarks)
                    if i in [61, 84, 17, 314, 405, 320, 307, 375, 321, 308]
                ]
            )

            if previous_mouth is not None:
                # Calculate motion between frames
                motion = np.mean(np.linalg.norm(mouth - previous_mouth, axis=1))
                lips_motion.append(motion)
            else:
                lips_motion.append(0)

            previous_mouth = mouth
        else:
            lips_motion.append(0)

print(f"Lip motion analysis complete. Processed {len(lips_motion)} frames.")
print(f"Average motion: {np.mean(lips_motion):.4f}")
print(f"Max motion: {max(lips_motion):.4f}")

# Detect speaking frames
speaking_threshold = 0.01
speaking_frames = [motion > speaking_threshold for motion in lips_motion]
speaking_percentage = sum(speaking_frames) / len(speaking_frames) * 100
print(f"Speaking detected in {speaking_percentage:.1f}% of frames")
