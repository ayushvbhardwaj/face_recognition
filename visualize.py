import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ===== CONFIG =====
BASE_DIR = "people"
MODEL_KEYS = ["average_cnn", "average_dct", "average_facenet"]
MODEL_COLORS = {"average_cnn": "red", "average_dct": "green", "average_facenet": "blue"}
MODEL_MARKERS = {"average_cnn": "o", "average_dct": "^", "average_facenet": "s"}

# ===== LOAD ALL EMBEDDINGS =====
embeddings = {key: [] for key in MODEL_KEYS}
labels = {key: [] for key in MODEL_KEYS}

for person in os.listdir(BASE_DIR):
    json_path = os.path.join(BASE_DIR, person, "embeddings.json")
    if not os.path.isfile(json_path):
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    for key in MODEL_KEYS:
        if key in data and len(data[key]) > 0:
            vec = np.array(data[key], dtype=np.float32)
            embeddings[key].append(vec)
            labels[key].append(person)

# ===== PLOT EACH MODEL SEPARATELY =====
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for key in MODEL_KEYS:
    if len(embeddings[key]) == 0:
        print(f"⚠️ No data found for {key}")
        continue

    X = np.stack(embeddings[key])  # (num_people, dim)
    n_samples = X.shape[0]

    print(f"Loaded {n_samples} embeddings for {key} ({X.shape[1]}D)")

    if n_samples >= 3:
        # True 3D PCA
        pca = PCA(n_components=3)
        X_proj = pca.fit_transform(X)
    else:
        # Not enough samples for true PCA → make fake 3D spread
        print(f"⚠️ {key}: Only {n_samples} samples — faking 3D layout.")
        X_proj = np.zeros((n_samples, 3))
        for i in range(n_samples):
            # scale and jitter slightly for visual separation
            X_proj[i] = np.random.randn(3) * 0.2 + i

    color = MODEL_COLORS[key]
    marker = MODEL_MARKERS[key]

    for i, person in enumerate(labels[key]):
        coords = X_proj[i]
        # --- draw 3D vector from origin ---
        ax.quiver(0, 0, 0, coords[0], coords[1], coords[2],
                  color=color, arrow_length_ratio=0.1, linewidth=1.5, alpha=0.8)

        # --- draw point ---
        ax.scatter(coords[0], coords[1], coords[2],
                   c=color, marker=marker, s=80,
                   label=f"{person} ({key.split('_')[1].upper()})")

        # --- text label ---
        ax.text(coords[0], coords[1], coords[2],
                f"{person[:3]}", fontsize=8, color=color)

ax.set_title("3D Projection of Average Embeddings (CNN / DCT / FaceNet)")
ax.set_xlabel("PC1 / X-axis")
ax.set_ylabel("PC2 / Y-axis")
ax.set_zlabel("PC3 / Z-axis")

# Make axes equal scale for better spatial visualization
max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).ptp().max() / 2.0
mid_x = np.mean(ax.get_xlim())
mid_y = np.mean(ax.get_ylim())
mid_z = np.mean(ax.get_zlim())
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Deduplicate legend
handles, labels_unique = ax.get_legend_handles_labels()
unique = dict(zip(labels_unique, handles))
ax.legend(unique.values(), unique.keys(), fontsize=9)

plt.tight_layout()
plt.show()