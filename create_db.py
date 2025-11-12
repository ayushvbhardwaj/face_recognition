import cv2
import os
import json
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp

def create_person_dataset(name):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
    
    # Initialize FaceNet
    model = InceptionResnetV1(pretrained='vggface2').eval()

    # Create folder for this person
    person_dir = f"people/{name}"
    os.makedirs(person_dir, exist_ok=True)

    # Webcam setup
    cap = cv2.VideoCapture(0)
    embeddings = []
    img_count = 0

    print(f"[INFO] Capturing images for '{name}'. Press 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)

                # Crop face safely
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = frame[y1:y2, x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Press 's' to save", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Dataset Creator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and results.detections:
            # Save cropped face
            img_path = os.path.join(person_dir, f"{name}_{img_count}.jpg")
            cv2.imwrite(img_path, face_crop)

            # Get embedding
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            face_tensor = torch.tensor(face_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                emb = model(face_tensor).numpy().flatten()
            emb = emb / np.linalg.norm(emb)

            embeddings.append(emb.tolist())
            img_count += 1

            print(f"[INFO] Saved {img_path}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Compute average embedding
    avg_embedding = np.mean(np.array(embeddings), axis=0).tolist() if embeddings else []

    # Save embeddings to JSON
    json_path = os.path.join(person_dir, "embeddings.json")
    data = {
        "name": name,
        "embeddings": embeddings,
        "average": avg_embedding
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[DONE] Saved {len(embeddings)} face(s) and embeddings for '{name}' in {person_dir}")


if __name__ == "__main__":
    create_person_dataset("ayush")