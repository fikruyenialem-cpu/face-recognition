import cv2
import os

save_folder = "owner_faces"

os.makedirs(save_folder, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Face capture started.")
print("Press SPACE to capture an image.")
print("Press Q inside the webcam window to quit.")

count = 0

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read webcam.")
        break

    cv2.putText(
        frame,
        f"Images captured: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture Owner Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        filename = os.path.join(
            save_folder,
            f"owner_{count:03d}.jpg"
        )

        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Finished. Captured {count} images.")