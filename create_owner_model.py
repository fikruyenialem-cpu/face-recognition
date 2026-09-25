import cv2
import insightface
import numpy as np
import os
import pickle

# --------------------------------
# Load InsightFace model
# --------------------------------
print("Loading face recognition model...")

app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("Model loaded!")

# --------------------------------
# Read owner images
# --------------------------------
folder = "owner_faces"

embeddings = []

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(files)} owner images.")

for filename in files:

    path = os.path.join(folder, filename)

    image = cv2.imread(path)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    # Detect faces
    faces = app.get(image)

    if len(faces) == 0:
        print(f"No face found: {filename}")
        continue

    # Use the largest face
    face = max(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) *
                      (x.bbox[3] - x.bbox[1])
    )

    # Get face embedding
    embedding = face.normed_embedding

    embeddings.append(embedding)

    print(f"Processed: {filename}")

# --------------------------------
# Check results
# --------------------------------
if len(embeddings) == 0:
    print("\nERROR: No usable faces were found.")
    exit()

# Average the embeddings
owner_embedding = np.mean(embeddings, axis=0)

# Normalize the final embedding
owner_embedding = owner_embedding / np.linalg.norm(owner_embedding)

# --------------------------------
# Save owner model
# --------------------------------
with open("owner_embedding.pkl", "wb") as file:
    pickle.dump(owner_embedding, file)

print("\n================================")
print("OWNER MODEL CREATED SUCCESSFULLY")
print("================================")
print(f"Usable images: {len(embeddings)}")
print("Saved as: owner_embedding.pkl")