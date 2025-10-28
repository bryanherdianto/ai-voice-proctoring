import cv2
import mediapipe as mp
import numpy as np
import time


class RealTimeLipDetector:
    def __init__(self, motion_threshold=0.002, speaking_delay=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Lip landmarks and reference points
        self.inner_lip_indices = [78, 81, 13, 82, 312, 311, 310, 415, 308, 324, 318]
        self.reference_indices = [1, 2, 5, 4, 6, 168, 8, 9, 10, 151]

        self.previous_lips = None
        self.previous_reference = None
        self.motion_threshold = motion_threshold
        self.speaking_delay = speaking_delay
        self.last_speaking_time = None
        self.current_speaking_status = False

    def extract_landmarks(self, landmarks, indices):
        points = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                points.append([lm.x, lm.y])
        return np.array(points)

    def calculate_relative_motion(
        self, current_lips, previous_lips, current_ref, previous_ref
    ):
        if previous_lips is None or previous_ref is None:
            return 0.0

        # Normalize lip positions relative to face reference centers
        current_ref_center = np.mean(current_ref, axis=0)
        previous_ref_center = np.mean(previous_ref, axis=0)

        current_lips_normalized = current_lips - current_ref_center
        previous_lips_normalized = previous_lips - previous_ref_center

        motion = np.mean(
            np.linalg.norm(current_lips_normalized - previous_lips_normalized, axis=1)
        )

        # Scale normalization
        current_scale = np.mean(
            np.linalg.norm(current_ref - current_ref_center, axis=1)
        )
        previous_scale = np.mean(
            np.linalg.norm(previous_ref - previous_ref_center, axis=1)
        )

        if previous_scale > 0:
            motion = motion / previous_scale

        return motion

    def calculate_mouth_metrics(self, lip_points):
        if len(lip_points) < 11:
            return 0.0, 0.0

        # Mouth openness (vertical distance)
        upper_center = lip_points[2]  # landmark 13
        lower_center = lip_points[8]  # landmark 308
        openness = abs(upper_center[1] - lower_center[1])

        # Mouth width (horizontal distance)
        left_corner = lip_points[0]  # landmark 78
        right_corner = lip_points[4]  # landmark 312
        width = abs(right_corner[0] - left_corner[0])

        # Aspect ratio
        aspect_ratio = openness / width if width > 0 else 0.0

        return openness, aspect_ratio

    def is_speaking_detected(self, motion, openness, aspect_ratio):
        mouth_open = openness > 0.015
        good_shape = aspect_ratio > 0.2
        has_motion = motion > self.motion_threshold

        return mouth_open and good_shape and has_motion

    def update_speaking_status(self, raw_speaking):
        current_time = time.time()

        if raw_speaking:
            self.current_speaking_status = True
            self.last_speaking_time = current_time
        else:
            if self.last_speaking_time is not None:
                time_since = current_time - self.last_speaking_time
                if time_since >= self.speaking_delay:
                    self.current_speaking_status = False
            else:
                self.current_speaking_status = False

        return self.current_speaking_status

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            current_lips = self.extract_landmarks(landmarks, self.inner_lip_indices)
            current_reference = self.extract_landmarks(
                landmarks, self.reference_indices
            )

            motion = 0.0
            openness = 0.0
            aspect_ratio = 0.0

            if self.previous_lips is not None:
                motion = self.calculate_relative_motion(
                    current_lips,
                    self.previous_lips,
                    current_reference,
                    self.previous_reference,
                )
                openness, aspect_ratio = self.calculate_mouth_metrics(current_lips)

            self.previous_lips = current_lips
            self.previous_reference = current_reference

            raw_speaking = self.is_speaking_detected(motion, openness, aspect_ratio)
            speaking_status = self.update_speaking_status(raw_speaking)

            return speaking_status
        else:
            speaking_status = self.update_speaking_status(False)
            return speaking_status


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    detector = RealTimeLipDetector(motion_threshold=0.002, speaking_delay=1.0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            speaking = detector.process_frame(frame)

            print("SPEAKING" if speaking else "silent")

            # Show clean camera feed
            cv2.imshow("Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        detector.face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
