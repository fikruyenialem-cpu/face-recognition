import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open the webcam.")
    exit()

print("Webcam started!")
print("Press Q inside the webcam window to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    cv2.imshow("My Webcam - OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()