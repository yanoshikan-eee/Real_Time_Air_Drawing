
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
print("Starting Gesture Recognition...")

start_time = time.perf_counter()
last_timestamp = -1


# Function to check whether a finger is extended
def is_finger_extended(hand_landmarks, tip, pip):

    return hand_landmarks[tip].y < hand_landmarks[pip].y


# Function to check whether thumb is extended
def is_thumb_extended(hand_landmarks):

    thumb_tip = hand_landmarks[4]
    thumb_mcp = hand_landmarks[2]
    index_mcp = hand_landmarks[5]

    # Calculate horizontal distances
    thumb_distance = abs(thumb_tip.x - index_mcp.x)

    base_distance = abs(thumb_mcp.x - index_mcp.x)

    # Check whether thumb is extended
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
                    3
                )

            # Draw landmarks
            for point in points:

                cv2.circle(
                    frame,
                    point,
                    6,
                    (0, 0, 255),
                    -1
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
            if finger_count == 0:

                gesture = "CLOSED FIST - STOP"
                color = (0, 0, 255)

            elif finger_count == 1 and index_up:

                gesture = "INDEX FINGER - DRAW"
                color = (0, 255, 0)

            elif finger_count == 2 and index_up and middle_up:

                gesture = "TWO FINGERS - MOVE"
                color = (255, 255, 0)

            elif finger_count == 5:

                gesture = "OPEN PALM - ALL FINGERS UP"
                color = (255, 0, 255)

            else:

                gesture = "OTHER GESTURE"
                color = (255, 255, 255)

            # Display recognized gesture
            cv2.putText(
                frame,
                gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Display raised finger count
            cv2.putText(
                frame,
                f"FINGERS: {finger_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Display output
        cv2.imshow(
            "Real-Time Air Drawing - Gesture Recognition",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Gesture Recognition stopped.")
