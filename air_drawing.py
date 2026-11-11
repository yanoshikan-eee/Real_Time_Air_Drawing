
import cv2
import mediapipe as mp
import numpy as np
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

print("Webcam connected!")
print("Starting Air Drawing System...")

start_time = time.perf_counter()
last_timestamp = -1

# Previous finger position
previous_x = 0
previous_y = 0

# Drawing control
drawing_enabled = True

# Smoothing factor
smoothing = 0.3

# Tracking region
FRAME_MARGIN_X = 100
FRAME_MARGIN_Y = 60

# Create drawing canvas after reading first frame
canvas = None

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

        # Create blank canvas
        if canvas is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
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

            # Get index fingertip
            index_tip = hand_landmarks[8]

            # Convert normalized coordinates to pixels
            current_x = int(index_tip.x * width)
            current_y = int(index_tip.y * height)

            # Restrict tracking region
            current_x = max(
                FRAME_MARGIN_X,
                min(current_x, width - FRAME_MARGIN_X)
            )

            current_y = max(
                FRAME_MARGIN_Y,
                min(current_y, height - FRAME_MARGIN_Y)
            )

            # Smooth cursor movement
            cursor_x = int(
                previous_x + (current_x - previous_x) * smoothing
            )

            cursor_y = int(
                previous_y + (current_y - previous_y) * smoothing
            )

            # Draw only when drawing is enabled
            if drawing_enabled:

                # Draw only when previous position exists
                if previous_x != 0 and previous_y != 0:

                    cv2.line(
                        canvas,
                        (previous_x, previous_y),
                        (cursor_x, cursor_y),
                        (0, 255, 0),
                        5
                    )

            # Update previous position
            previous_x = cursor_x
            previous_y = cursor_y

            # Draw fingertip cursor
            cv2.circle(
                frame,
                (cursor_x, cursor_y),
                10,
                (255, 0, 0),
                -1
            )

            # Display drawing status
            if drawing_enabled:
                drawing_status = "DRAWING: ON"
            else:
                drawing_status = "DRAWING: OFF"

            cv2.putText(
                frame,
                drawing_status,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if drawing_enabled else (0, 0, 255),
                2
            )

            # Display tracking status
            cv2.putText(
                frame,
                "AIR DRAWING ACTIVE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            # Reset previous position
            previous_x = 0
            previous_y = 0

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Draw tracking region
        cv2.rectangle(
            frame,
            (FRAME_MARGIN_X, FRAME_MARGIN_Y),
            (width - FRAME_MARGIN_X, height - FRAME_MARGIN_Y),
            (0, 255, 0),
            2
        )

        # Combine webcam and drawing canvas
        output = cv2.addWeighted(
            frame, 1,
            canvas, 1,
            0
        )

        # Display final output
        cv2.imshow(
            "Real-Time Air Drawing System",
            output
        )

        # Press C to clear canvas
        key = cv2.waitKey(1) & 0xFF

        # Press SPACE to enable or disable drawing
        if key == ord(' '):
            drawing_enabled = not drawing_enabled

            # Reset previous position
            previous_x = 0
            previous_y = 0

        if key == ord('c'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            previous_x = 0
            previous_y = 0

        # Press Q to exit
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Air Drawing System stopped.")
