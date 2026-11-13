
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

print("3D Air Drawing Started!")
print("Index finger: DRAW")
print("Index + Middle fingers: MOVE")
print("Closed fist: STOP")
print("Press C: Clear drawing")
print("Press Q: Quit")

# Store separate 3D drawing strokes
drawing_strokes = []

# Current stroke
current_stroke = None

# Previous point
previous_point = None

# Depth scale
DEPTH_SCALE = 2.0

# Initialize timestamp
start_time = time.perf_counter()
last_timestamp = -1

# Initialize 3D plot
plt.ion()

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Real-Time 3D Air Drawing")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Store separate 3D line objects
stroke_lines = []


# Function to check whether a finger is extended
def is_finger_extended(hand_landmarks, tip, pip):

    return hand_landmarks[tip].y < hand_landmarks[pip].y


# Function to clear 3D drawing
def clear_drawing():

    global previous_point, current_stroke

    drawing_strokes.clear()
    previous_point = None
    current_stroke = None

    # Remove existing 3D lines
    for stroke_line in stroke_lines:
        stroke_line.remove()

    stroke_lines.clear()

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    print("3D Drawing Cleared")


# Handle keyboard input from Matplotlib 3D viewer
def handle_3d_key(event):

    if event.key in ("c", "C"):
        clear_drawing()


# Connect keyboard event to 3D viewer
fig.canvas.mpl_connect(
    "key_press_event",
    handle_3d_key
)


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

            hand_landmarks = result.hand_landmarks[0]

            # Detect extended fingers
            index_up = is_finger_extended(
                hand_landmarks, 8, 6
            )

            middle_up = is_finger_extended(
                hand_landmarks, 12, 10
            )

            ring_up = is_finger_extended(
                hand_landmarks, 16, 14
            )

            little_up = is_finger_extended(
                hand_landmarks, 20, 18
            )

            # Count extended fingers
            finger_count = sum([
                index_up,
                middle_up,
                ring_up,
                little_up
            ])

            # Recognize gestures
            if index_up and not middle_up:

                gesture = "DRAW"
                gesture_color = (0, 255, 0)

            elif index_up and middle_up:

                gesture = "MOVE"
                gesture_color = (255, 255, 0)

            elif finger_count == 0:

                gesture = "STOP"
                gesture_color = (0, 0, 255)

            else:

                gesture = "OTHER GESTURE"
                gesture_color = (255, 255, 255)

            # Get index fingertip
            index_tip = hand_landmarks[8]

            # Get normalized 3D coordinates
            x = index_tip.x
            y = index_tip.y
            z = index_tip.z

            # Convert coordinates to 3D drawing space
            point_3d = (
                (x - 0.5) * 2,
                (0.5 - y) * 2,
                -z * DEPTH_SCALE
            )

            # Convert to webcam coordinates
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

            # Display coordinates
            cv2.putText(
                frame,
                f"X: {x:.3f} Y: {y:.3f} Z: {z:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Day 19: Gesture-controlled 3D drawing
            if gesture == "DRAW":

                # Start a new stroke if needed
                if previous_point is None:

                    current_stroke = []
                    drawing_strokes.append(current_stroke)

                    # Create a separate 3D line
                    new_line, = ax.plot(
                        [], [], [],
                        linewidth=2
                    )

                    stroke_lines.append(new_line)

                # Calculate movement distance
                if previous_point is not None:

                    movement_distance = np.linalg.norm(
                        np.array(point_3d) - np.array(previous_point)
                    )

                    if movement_distance >= 0.30:

                        # Start a separate stroke
                        current_stroke = []
                        drawing_strokes.append(current_stroke)

                        new_line, = ax.plot(
                            [], [], [],
                            linewidth=2
                        )

                        stroke_lines.append(new_line)

                # Add current point
                current_stroke.append(point_3d)

                previous_point = point_3d

            # MOVE gesture
            elif gesture == "MOVE":

                previous_point = None
                current_stroke = None

            # STOP gesture
            elif gesture == "STOP":

                previous_point = None
                current_stroke = None

            # Other gestures
            else:

                previous_point = None
                current_stroke = None

            # Display recognized gesture
            cv2.putText(
                frame,
                "GESTURE: " + gesture,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                gesture_color,
                2
            )

        else:

            # Pause drawing when no hand is detected
            previous_point = None
            current_stroke = None

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Update 3D viewer
        if drawing_strokes:

            for i, stroke in enumerate(drawing_strokes):

                if len(stroke) > 0:

                    points_array = np.array(stroke)

                    stroke_lines[i].set_data(
                        points_array[:, 0],
                        points_array[:, 1]
                    )

                    stroke_lines[i].set_3d_properties(
                        points_array[:, 2]
                    )

            # Adjust 3D viewing limits
            all_points = [
                point
                for stroke in drawing_strokes
                for point in stroke
            ]

            if all_points:

                points_array = np.array(all_points)

                margin = 0.2

                ax.set_xlim(
                    points_array[:, 0].min() - margin,
                    points_array[:, 0].max() + margin
                )

                ax.set_ylim(
                    points_array[:, 1].min() - margin,
                    points_array[:, 1].max() + margin
                )

                ax.set_zlim(
                    points_array[:, 2].min() - margin,
                    points_array[:, 2].max() + margin
                )

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        # Display webcam
        cv2.imshow(
            "3D Air Drawing - Hand Tracking",
            frame
        )

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('c'), ord('C')):

            clear_drawing()

        elif key in (ord('q'), ord('Q')):

            break

        # Stop if the 3D viewer is closed
        if not plt.fignum_exists(fig.number):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.close('all')

print("3D Air Drawing Stopped.")
