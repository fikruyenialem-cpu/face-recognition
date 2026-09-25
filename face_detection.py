import cv2

# -----------------------------
# Load YuNet face detector
# -----------------------------
model = "face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    model,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

# -----------------------------
# Open the PC webcam
# -----------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("YuNet face detection started!")
print("Press Q inside the webcam window to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # Get current frame size
    height, width = frame.shape[:2]

    # Tell YuNet the size of the image
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    # Draw results
    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Face confidence
            confidence = face[-1]

            text = f"Face: {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Show webcam
    cv2.imshow("YuNet Face Detection", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()