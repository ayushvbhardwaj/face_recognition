import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
import os, json
from embeddings import cnn_embedding, dct_embedding

# ───────────────────────────────────────────────
# 🔧 GLOBAL SETTINGS
# ───────────────────────────────────────────────
MODEL_MODE = "facenet"      # "cnn", "dct", "facenet"
USE_ALL_MODELS = True       # if True, shows all 3 on-screen

# Initialize FaceNet once
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()


# ───────────────────────────────────────────────
# 🧠 GET EMBEDDING FUNCTION
# ───────────────────────────────────────────────
def get_embedding(face_crop, model_name):
    if face_crop is None or face_crop.size == 0:
        return None

    model_name = model_name.lower()
    if model_name == "facenet":
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        tensor = torch.tensor(face_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            emb = facenet_model(tensor).numpy().flatten()

    elif model_name == "cnn":
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_crop)
        emb = cnn_embedding(temp_path)

    elif model_name == "dct":
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_crop)
        emb = dct_embedding(temp_path)
    else:
        return None

    emb = np.array(emb)
    emb = emb / np.linalg.norm(emb)
    return emb


# ───────────────────────────────────────────────
# 📂 LOAD PEOPLE EMBEDDINGS
# ───────────────────────────────────────────────
def load_people_embeddings(base_dir="people"):
    people_embeddings = {"cnn": {}, "dct": {}, "facenet": {}}

    for person_name in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_name)
        json_path = os.path.join(person_path, "embeddings.json")
        if not os.path.isfile(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        for model_type in ["cnn", "dct", "facenet"]:
            key = f"average_{model_type}"
            if key in data and data[key]:
                people_embeddings[model_type][person_name] = np.array(data[key], dtype=np.float32)

    return people_embeddings


# ───────────────────────────────────────────────
# 🔍 CALCULATE DOT PRODUCT
# ───────────────────────────────────────────────
def calculate_dot_product(people_embeddings, face_embedding):
    if face_embedding is None:
        return None, 0.0
    scores = {person: float(np.dot(emb, face_embedding)) for person, emb in people_embeddings.items()}
    if not scores:
        return None, 0.0
    best_person = max(scores, key=scores.get)
    best_score = scores[best_person]
    return best_person, best_score


# ───────────────────────────────────────────────
# 🎥 MAIN LOOP (Continuous Recognition)
# ───────────────────────────────────────────────
def main():
    print(f"🧠 Mode: {MODEL_MODE.upper()} | USE_ALL_MODELS={USE_ALL_MODELS}")
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    all_embeddings = load_people_embeddings()

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                         int(bboxC.width * w), int(bboxC.height * h)
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + w_box), min(h, y + h_box)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Predict with selected model(s)
                    predictions = {}
                    if USE_ALL_MODELS:
                        for model_type in ["cnn", "dct", "facenet"]:
                            emb = get_embedding(face_crop, model_type)
                            person, score = calculate_dot_product(all_embeddings[model_type], emb)
                            predictions[model_type] = (person or "Unknown", round(score, 3))
                    else:
                        emb = get_embedding(face_crop, MODEL_MODE)
                        person, score = calculate_dot_product(all_embeddings[MODEL_MODE], emb)
                        predictions[MODEL_MODE] = (person or "Unknown", round(score, 3))

                    # Display predictions above the bounding box
                    y_offset = y1 - 10
                    for model_type, (person, score) in predictions.items():
                        text = f"{model_type.upper()}: {person} ({score:.2f})"
                        color = (0, 255, 0) if score > 0.6 else (0, 0, 255)
                        cv2.putText(frame, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset -= 25

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ───────────────────────────────────────────────
# 🚀 RUN
# ───────────────────────────────────────────────
if __name__ == "__main__":
    main()