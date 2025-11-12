import os
import json
import numpy as np
from embeddings import cnn_embedding, dct_embedding
from facenet_pytorch import InceptionResnetV1
import torch
import cv2
from tqdm import tqdm

# ───────────────────────────────────────────────
# INITIALIZE FACENET ONCE
# ───────────────────────────────────────────────
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

def get_facenet_embedding(image):
    """Returns 512D FaceNet embedding for an image."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        emb = facenet_model(tensor).numpy().flatten()
    return emb / np.linalg.norm(emb)

# ───────────────────────────────────────────────
# MAIN FUNCTION
# ───────────────────────────────────────────────
def generate_embeddings(folder):
    """
    For each person folder, compute:
    - Average CNN embedding (128D)
    - Average DCT embedding (128D)
    - Average FaceNet embedding (512D)
    and store only these in embeddings.json.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"❌ Folder '{folder}' not found!")

    all_cnn, all_dct, all_facenet = [], [], []

    image_files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )
    if not image_files:
        print(f"⚠️ No valid images in {folder}")
        return

    print(f"📂 Folder: {folder}\n📸 Found {len(image_files)} images.")

    for fname in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {fname}")
            continue

        try:
            cnn_vec = cnn_embedding(img_path)
            dct_vec = dct_embedding(img_path)
            facenet_vec = get_facenet_embedding(img)

            all_cnn.append(cnn_vec)
            all_dct.append(dct_vec)
            all_facenet.append(facenet_vec)
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    if not all_cnn:
        print("❌ No embeddings generated — aborting.")
        return


    avg_cnn = np.mean(all_cnn, axis=0).tolist()
    avg_dct = np.mean(all_dct, axis=0).tolist()
    avg_facenet = np.mean(all_facenet, axis=0).tolist()

    output = {
        "average_cnn": avg_cnn,
        "average_dct": avg_dct,
        "average_facenet": avg_facenet
    }

    out_path = os.path.join(folder, "embeddings.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved 3 averaged embeddings to: {out_path}")
    print(f"🧠 CNN: {len(avg_cnn)}D | DCT: {len(avg_dct)}D | FaceNet: {len(avg_facenet)}D")
    print("✅ Done!")

# ───────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate averaged embeddings for a folder.")
    parser.add_argument("folder", type=str, help="Path to the person's folder")
    args = parser.parse_args()
    generate_embeddings(args.folder)