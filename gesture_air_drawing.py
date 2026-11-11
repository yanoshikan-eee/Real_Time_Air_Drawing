
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
print("Starting Gesture-Controlled Air Drawing...")

start_time = time.perf_counter()
last_timestamp = -1

# Previous finger position
previous_x = 0
previous_y = 0

# Smoothed cursor position
smooth_x = 0
smooth_y = 0

# Previous gesture
previous_gesture = None

# Tracking region
FRAME_MARGIN_X = 100
FRAME_MARGIN_Y = 60

# Create drawing canvas
canvas = None

# Selected drawing colour (BGR format)
drawing_color = (0, 255, 0)


# Function to check whether a finger is extended
def is_finger_extended(hand_landmarks, tip, pip):

    return hand_landmarks[tip].y < hand_landmarks[pip].y


# Function to check whether thumb is extended
def is_thumb_extended(hand_landmarks):

    thumb_tip = hand_landmarks[4]
    thumb_mcp = hand_landmarks[2]
    index_mcp = hand_landmarks[5]

    thumb_distance = abs(thumb_tip.x - index_mcp.x)
    base_distance = abs(thumb_mcp.x - index_mcp.x)

    return thumb_distance > base_distance + 0.04


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

            # Convert landmarks to pixel coordinates
            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

            # Draw hand connections
            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            # Detect extended fingers
            thumb_up = is_thumb_extended(hand_landmarks)

            index_up = is_finger_extended(hand_landmarks, 8, 6)
            middle_up = is_finger_extended(hand_landmarks, 12, 10)
            ring_up = is_finger_extended(hand_landmarks, 16, 14)
            little_up = is_finger_extended(hand_landmarks, 20, 18)

            # Count raised fingers
            finger_count = sum([
                thumb_up,
                index_up,
                middle_up,
                ring_up,
                little_up
            ])

            # Recognize gestures
            if index_up and not middle_up and finger_count < 5:

                gesture = "DRAW"
                color = (0, 255, 0)

            elif finger_count == 2 and index_up and middle_up:

                gesture = "MOVE"
                color = (255, 255, 0)

            elif finger_count == 0:

                gesture = "STOP"
                color = (0, 0, 255)

            elif finger_count == 5:

                gesture = "CLEAR"
                color = (255, 0, 255)

            else:

                gesture = "OTHER GESTURE"
                color = (255, 255, 255)

            # Get index fingertip coordinates
            index_tip = hand_landmarks[8]

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

            # Adaptive cursor smoothing
            if smooth_x == 0 and smooth_y == 0:

                smooth_x = current_x
                smooth_y = current_y

            else:

                # Calculate cursor movement distance
                distance = np.hypot(
                    current_x - smooth_x,
                    current_y - smooth_y
                )

                # Adjust smoothing according to movement
                if distance > 8:
                    adaptive_smoothing = 0.95
                else:
                    adaptive_smoothing = 0.65

                # Smooth cursor position
                smooth_x = int(
                    smooth_x + (current_x - smooth_x) * adaptive_smoothing
                )

                smooth_y = int(
                    smooth_y + (current_y - smooth_y) * adaptive_smoothing
                )

            cursor_x = smooth_x
            cursor_y = smooth_y

            # DRAW gesture
            if gesture == "DRAW":

                if previous_x != 0 and previous_y != 0:

                    cv2.line(
                        canvas,
                        (previous_x, previous_y),
                        (cursor_x, cursor_y),
                        drawing_color,
                        5,
                        cv2.LINE_AA
                    )

                previous_x = cursor_x
                previous_y = cursor_y

            # MOVE gesture
            elif gesture == "MOVE":

                previous_x = 0
                previous_y = 0

            # STOP gesture
            elif gesture == "STOP":

                previous_x = 0
                previous_y = 0

            # CLEAR gesture
            elif gesture == "CLEAR":

                if previous_gesture != "CLEAR":

                    canvas = np.zeros(
                        (height, width, 3),
                        dtype=np.uint8
                    )

                previous_x = 0
                previous_y = 0

            # Other gestures
            else:

                previous_x = 0
                previous_y = 0

            # Update previous gesture
            previous_gesture = gesture

            # Draw fingertip cursor
            cv2.circle(
                frame,
                (cursor_x, cursor_y),
                10,
                (255, 0, 0),
                -1
            )

            # Display recognized gesture
            cv2.putText(
                frame,
                "GESTURE: " + gesture,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Display finger count
            cv2.putText(
                frame,
                f"FINGERS: {finger_count}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            # Reset previous position and gesture
            previous_x = 0
            previous_y = 0
            smooth_x = 0
            smooth_y = 0
            previous_gesture = None

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
            "Gesture-Controlled Air Drawing System",
            output
        )

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        # Colour selection
        if key == ord('r'):
            drawing_color = (0, 0, 255)

        elif key == ord('g'):
            drawing_color = (0, 255, 0)

        elif key == ord('b'):
            drawing_color = (255, 0, 0)

        elif key == ord('y'):
            drawing_color = (0, 255, 255)

        elif key == ord('w'):
            drawing_color = (255, 255, 255)

        elif key == ord('k'):
            drawing_color = (0, 0, 0)

        # Press C to clear canvas manually
        if key == ord('c'):

            canvas = np.zeros(
                (height, width, 3),
                dtype=np.uint8
            )

            previous_x = 0
            previous_y = 0

        # Press Q to exit
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Gesture-Controlled Air Drawing stopped.")
