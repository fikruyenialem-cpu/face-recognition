import cv2
import insightface
import numpy as np
import pickle
import serial
import time


# =========================
# SETTINGS
# =========================

SERIAL_PORT = "COM13"
BAUD_RATE = 9600
THRESHOLD = 0.60


# =========================
# CONNECT ARDUINO
# =========================

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0.1
)

time.sleep(2)

print("================================")
print("ARDUINO CONNECTED: COM14")
print("================================")


# =========================
# LOAD OWNER
# =========================

with open("owner_embedding.pkl", "rb") as file:
    owner_embedding = pickle.load(file)


# =========================
# INSIGHTFACE
# =========================

app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("ERROR: Webcam could not open.")

    arduino.close()
    exit()


print("WEBCAM READY")
print("SHOW YOUR FACE")
print("Q = QUIT")


# =========================
# STATE
# =========================

ready_for_face = False

last_bad_face = 0

BAD_FACE_DELAY = 3


# =========================
# MAIN LOOP
# =========================

while True:

    # =========================
    # READ ARDUINO
    # =========================

    while arduino.in_waiting:

        message = arduino.readline().decode(
            errors="ignore"
        ).strip()

        if message:

            print("Arduino:", message)


            if message == "READY_FACE":

                ready_for_face = True

                print(">>> SHOW FACE <<<")


            elif message == "FACE_VERIFIED":

                ready_for_face = False


            elif message == "ENTER_P1":

                ready_for_face = False

                print(">>> ENTER P1 <<<")


            elif message == "ENTER_P2":

                ready_for_face = False

                print(">>> ENTER P2 <<<")


    # =========================
    # CAMERA
    # =========================

    success, frame = camera.read()

    if not success:
        continue


    faces = app.get(frame)


    # =========================
    # FACE RECOGNITION
    # =========================

    for face in faces:

        embedding = face.normed_embedding

        similarity = np.dot(
            owner_embedding,
            embedding
        )

        x1, y1, x2, y2 = face.bbox.astype(int)


        # =========================
        # OWNER
        # =========================

        if similarity >= THRESHOLD:

            text = f"OWNER {similarity:.2f}"


            if ready_for_face:

                print()
                print("OWNER VERIFIED")

                arduino.write(
                    b"FACE_OK\n"
                )

                ready_for_face = False


        # =========================
        # UNKNOWN
        # =========================

        else:

            text = f"UNKNOWN {similarity:.2f}"


            if ready_for_face:

                if time.time() - last_bad_face >= BAD_FACE_DELAY:

                    print()
                    print("UNKNOWN FACE")

                    arduino.write(
                        b"FACE_BAD\n"
                    )

                    last_bad_face = time.time()


        # =========================
        # DRAW
        # =========================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # =========================
    # DISPLAY
    # =========================

    cv2.imshow(
        "FACE RECOGNITION DOOR LOCK",
        frame
    )


    # =========================
    # Q = QUIT
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

camera.release()
arduino.close()
cv2.destroyAllWindows()

print("SYSTEM STOPPED")