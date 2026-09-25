import cv2
import insightface
import numpy as np
import pickle
import serial
import time

SERIAL_PORT = "COM14"
BAUD_RATE = 9600
THRESHOLD = 0.60

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("Arduino connected on COM14")
print("Starting face recognition...")
print("Press Q to quit.")

with open("owner_embedding.pkl", "rb") as file:
    owner_embedding = pickle.load(file)

app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    arduino.close()
    exit()

last_unlock_time = 0

while True:
    success, frame = camera.read()

    if not success:
        break

    faces = app.get(frame)
    current_time = time.time()

    for face in faces:
        current_embedding = face.normed_embedding

        similarity = np.dot(
            owner_embedding,
            current_embedding
        )

        x1, y1, x2, y2 = face.bbox.astype(int)

        if similarity >= THRESHOLD:
            text = f"OWNER {similarity:.2f}"

            if current_time - last_unlock_time > 8:
                print(f"OWNER DETECTED! Score: {similarity:.2f}")
                print("Sending UNLOCK to Arduino...")

                arduino.write(b"UNLOCK\n")
                last_unlock_time = current_time

        else:
            text = f"UNKNOWN {similarity:.2f}"

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )

    cv2.imshow("FACE RECOGNITION DOOR LOCK", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
arduino.close()
cv2.destroyAllWindows()

print("System stopped.")