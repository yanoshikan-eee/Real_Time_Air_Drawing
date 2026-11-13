
import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf

# Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand Landmarker model location
MODEL_PATH = "models/hand_landmarker.task"

# Load handwriting recognition model
handwriting_model = tf.keras.models.load_model(
    "handwriting_model.keras"
)

with open("handwriting_labels.txt", "r") as file:
    handwriting_labels = file.read().strip()

print("Handwriting recognition model loaded successfully!")

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

# Gesture stabilization
candidate_gesture = None
candidate_start_time = 0
GESTURE_HOLD_TIME = 0.3

# Tracking region
FRAME_MARGIN_X = 100
FRAME_MARGIN_Y = 60

# Create drawing canvas
canvas = None

# Selected drawing colour (BGR format)
drawing_color = (0, 255, 0)

# Brush thickness
brush_thickness = 5

# Eraser mode
eraser_mode = False

# Eraser thickness
eraser_thickness = 25

# Handwriting recognition result
recognition_result = None
recognition_confidence = 0.0


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


# Function to prepare drawing for handwriting recognition
def prepare_handwriting_image(canvas):

    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        20,
        255,
        cv2.THRESH_BINARY
    )

    coordinates = cv2.findNonZero(binary)

    if coordinates is None:
        return None

    x, y, w, h = cv2.boundingRect(coordinates)

    cropped = binary[y:y + h, x:x + w]

    # Preserve aspect ratio
    scale = 20.0 / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Center character in 28 x 28 image
    processed = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    processed[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    processed = processed.astype("float32") / 255.0

    processed = processed.reshape(1, 28, 28, 1)

    return processed


# Function to recognize handwritten character
def recognize_handwriting(canvas):

    global recognition_result
    global recognition_confidence

    processed_image = prepare_handwriting_image(canvas)

    if processed_image is None:

        recognition_result = "NO DRAWING"
        recognition_confidence = 0.0

        print("No drawing available for recognition.")
        return

    # Match the original EMNIST dataset orientation
    processed_image = np.transpose(
        processed_image,
        (0, 2, 1, 3)
    )

    # Display exact 28 x 28 image sent to CNN
    display_image = processed_image[0].reshape(28, 28)

    cv2.imshow(
        "Image Sent to CNN",
        display_image
    )

    # Predict character
    predictions = handwriting_model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    recognition_confidence = float(
        np.max(predictions[0]) * 100
    )

    recognition_result = handwriting_labels[predicted_index]

    print("Recognized Character:", recognition_result)
    print("Confidence:", round(recognition_confidence, 2), "%")


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
            if thumb_up and not index_up and not middle_up and not ring_up and not little_up:

                gesture = "BRUSH_UP"
                color = (0, 255, 255)

            elif thumb_up and index_up and not middle_up and not ring_up and not little_up:

                gesture = "BRUSH_DOWN"
                color = (0, 255, 255)

            elif thumb_up and index_up and middle_up and ring_up and not little_up:

                gesture = "ERASER_ON"
                color = (0, 0, 255)

            elif thumb_up and index_up and middle_up and not ring_up and little_up:

                gesture = "DRAW_MODE"
                color = (0, 255, 0)

            elif index_up and not middle_up and finger_count < 5:

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

            # Stabilize advanced gestures
            advanced_gestures = [
                "BRUSH_UP",
                "BRUSH_DOWN",
                "ERASER_ON",
                "DRAW_MODE"
            ]

            if gesture in advanced_gestures:

                if candidate_gesture != gesture:
                    candidate_gesture = gesture
                    candidate_start_time = time.perf_counter()

                if time.perf_counter() - candidate_start_time < GESTURE_HOLD_TIME:
                    gesture = "OTHER GESTURE"
                    color = (255, 255, 255)

            else:
                candidate_gesture = None
                candidate_start_time = 0

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

                distance = np.hypot(
                    current_x - smooth_x,
                    current_y - smooth_y
                )

                if distance > 8:
                    adaptive_smoothing = 0.95
                else:
                    adaptive_smoothing = 0.65

                smooth_x = int(
                    smooth_x + (current_x - smooth_x) * adaptive_smoothing
                )

                smooth_y = int(
                    smooth_y + (current_y - smooth_y) * adaptive_smoothing
                )

            cursor_x = smooth_x
            cursor_y = smooth_y

            # Advanced gesture actions
            if gesture == "BRUSH_UP":

                if previous_gesture != "BRUSH_UP":
                    brush_thickness = min(brush_thickness + 1, 30)
                    print("Brush thickness:", brush_thickness)

                previous_x = 0
                previous_y = 0

            elif gesture == "BRUSH_DOWN":

                if previous_gesture != "BRUSH_DOWN":
                    brush_thickness = max(brush_thickness - 1, 1)
                    print("Brush thickness:", brush_thickness)

                previous_x = 0
                previous_y = 0

            elif gesture == "ERASER_ON":

                if previous_gesture != "ERASER_ON":
                    eraser_mode = True
                    print("Eraser mode enabled by gesture")

                previous_x = 0
                previous_y = 0

            elif gesture == "DRAW_MODE":

                if previous_gesture != "DRAW_MODE":
                    eraser_mode = False
                    print("Normal drawing mode enabled by gesture")

                previous_x = 0
                previous_y = 0

            # DRAW gesture
            elif gesture == "DRAW":

                if previous_x != 0 and previous_y != 0:

                    movement_distance = np.hypot(
                        cursor_x - previous_x,
                        cursor_y - previous_y
                    )

                    if movement_distance < 80:

                        if eraser_mode:
                            current_color = (0, 0, 0)
                            current_thickness = eraser_thickness

                        else:
                            current_color = drawing_color
                            current_thickness = brush_thickness

                        cv2.line(
                            canvas,
                            (previous_x, previous_y),
                            (cursor_x, cursor_y),
                            current_color,
                            current_thickness,
                            cv2.LINE_AA
                        )

                        cv2.circle(
                            canvas,
                            (cursor_x, cursor_y),
                            max(1, current_thickness // 2),
                            current_color,
                            -1,
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

                    recognition_result = None
                    recognition_confidence = 0.0

                previous_x = 0
                previous_y = 0

            else:

                previous_x = 0
                previous_y = 0

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

            previous_x = 0
            previous_y = 0
            smooth_x = 0
            smooth_y = 0
            previous_gesture = None

            candidate_gesture = None
            candidate_start_time = 0

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Display brush thickness
        cv2.putText(
            frame,
            f"BRUSH SIZE: {brush_thickness}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display drawing mode
        if eraser_mode:
            mode_text = "MODE: ERASER"
            mode_color = (0, 0, 255)

        else:
            mode_text = "MODE: DRAWING"
            mode_color = (0, 255, 0)

        cv2.putText(
            frame,
            mode_text,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            mode_color,
            2
        )

        # Display handwriting recognition result
        if recognition_result is not None:

            recognition_text = (
                f"RECOGNIZED: {recognition_result} "
                f"({recognition_confidence:.1f}%)"
            )

            cv2.putText(
                frame,
                recognition_text,
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
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

        # Handwriting recognition
        if key in (ord('h'), ord('H')):
            recognize_handwriting(canvas)

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

        # Increase brush thickness
        elif key in (ord('+'), ord('=')):
            brush_thickness = min(brush_thickness + 1, 30)
            print("Brush thickness:", brush_thickness)

        # Decrease brush thickness
        elif key == ord('-'):
            brush_thickness = max(brush_thickness - 1, 1)
            print("Brush thickness:", brush_thickness)

        # Enable eraser mode
        elif key in (ord('e'), ord('E')):
            eraser_mode = True
            previous_x = 0
            previous_y = 0
            print("Eraser mode enabled")

        # Return to normal drawing mode
        elif key in (ord('n'), ord('N')):
            eraser_mode = False
            previous_x = 0
            previous_y = 0
            print("Normal drawing mode enabled")

        # Press C to clear canvas manually
        if key == ord('c'):

            canvas = np.zeros(
                (height, width, 3),
                dtype=np.uint8
            )

            previous_x = 0
            previous_y = 0

            recognition_result = None
            recognition_confidence = 0.0

        # Press Q to exit
        if key == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Gesture-Controlled Air Drawing stopped.")
