
import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model location
MODEL_PATH = "models/hand_landmarker.task"

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Configure hand detection
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

print("Webcam connected!")
print("Starting hand detection...")

start_time = time.perf_counter()
last_timestamp = -1

# Initialize hand landmarker
with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive video!")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate increasing timestamp
        timestamp_ms = max(
            int((time.perf_counter() - start_time) * 1000),
            last_timestamp + 1
        )
        last_timestamp = timestamp_ms

        # Detect hand landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Check whether hands are detected
        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                height, width, _ = frame.shape
                points = []

                # Convert normalized coordinates to pixels
                for landmark in hand_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                # Draw connections
                for start, end in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        3
                    )

                # Draw 21 landmarks
                for index, point in enumerate(points):

                    cv2.circle(
                        frame,
                        point,
                        6,
                        (0, 0, 255),
                        -1
                    )

                # Highlight index fingertip (Landmark 8)
                cv2.circle(
                    frame,
                    points[8],
                    12,
                    (255, 0, 0),
                    -1
                )

            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display output
        cv2.imshow(
            "Real-Time Air Drawing - Hand Tracking",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam
cap.release()
cv2.destroyAllWindows()

print("Hand tracking stopped.")
