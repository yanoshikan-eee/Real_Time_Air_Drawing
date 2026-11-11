
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check camera connection
if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

print("Webcam connected successfully!")

while True:

    # Capture video frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot receive video!")
        break

    # Mirror the video
    frame = cv2.flip(frame, 1)

    # Display video
    cv2.imshow("Real-Time Air Drawing - Camera Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()
