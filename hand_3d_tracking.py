
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

# Configure hand detection
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

print("3D Hand Tracking Started!")
print("Press Q to exit.")

start_time = time.perf_counter()
last_timestamp = -1

# Initialize hand detector
with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive video!")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate timestamp
        timestamp_ms = max(
            int((time.perf_counter() - start_time) * 1000),
            last_timestamp + 1
        )

        last_timestamp = timestamp_ms

        # Detect hand
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.hand_landmarks:

            # Get first detected hand
            hand_landmarks = result.hand_landmarks[0]

            # Get index fingertip (landmark 8)
            index_tip = hand_landmarks[8]

            # Extract 3D coordinates
            x = index_tip.x
            y = index_tip.y
            z = index_tip.z

            # Convert X and Y to screen coordinates
            screen_x = int(x * width)
            screen_y = int(y * height)

            # Draw fingertip marker
            cv2.circle(
                frame,
                (screen_x, screen_y),
                10,
                (0, 255, 0),
                -1
            )

            # Display 3D coordinates
            cv2.putText(
                frame,
                f"X: {x:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Y: {y:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Z: {z:.3f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "3D HAND TRACKING ACTIVE",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Display output
        cv2.imshow(
            "3D Hand Tracking",
            frame
        )

        # Press Q to exit
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("3D Hand Tracking Stopped.")
