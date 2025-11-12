import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy.fftpack import dct
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# 🔹 DCT EMBEDDING
# =============================
def dct_embedding(img_path, embed_dim=128, size=(128, 128)):
    """Compute a purely mathematical 128D embedding using 2D DCT."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    img = cv2.resize(img, size)

    # Apply 2D DCT
    dct_coeff = dct(dct(img.T, norm='ortho').T, norm='ortho')

    # Extract top-left low-frequency coefficients
    block_size = int(np.ceil(np.sqrt(embed_dim)))
    dct_crop = dct_coeff[:block_size, :block_size].flatten()[:embed_dim]

    # Normalize
    dct_vec = dct_crop / np.linalg.norm(dct_crop)
    return dct_vec


def compute_dct_folder_std(image_folder, embed_dim=128):
    """Compute DCT embeddings and their average std deviation across a folder."""
    embeddings = []
    filenames = []

    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(image_folder, fname)
        try:
            emb = dct_embedding(path, embed_dim=embed_dim)
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")

    if len(embeddings) < 2:
        raise ValueError("Need at least 2 valid images to compute std deviation.")

    embeddings = np.stack(embeddings)
    mean_vec = np.mean(embeddings, axis=0)
    std_vec = np.std(embeddings, axis=0)
    avg_std = np.mean(std_vec)

    print("\n📊 DCT Embedding Stats")
    print(f"Computed on {len(embeddings)} images")
    print(f"Average std deviation across DCT embedding dims: {avg_std:.5f}")
    print("First 10 elements of mean vector:", mean_vec[:10])
    print("First 10 elements of std vector:", std_vec[:10])
    print(f"Overall average std deviation: {avg_std:.5f}\n")

    return mean_vec, std_vec, avg_std


# =============================
# 🔹 PCA EMBEDDING
# =============================
def compute_pca_embeddings(image_folder, n_components=128, image_size=(128, 128)):
    """Compute PCA embeddings for all images in a folder."""
    images, filenames = [], []

    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(image_folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read: {path}")
            continue
        img = cv2.resize(img, image_size)
        images.append(img.flatten())
        filenames.append(fname)

    if len(images) == 0:
        raise ValueError("No valid images found in the folder!")

    X = np.stack(images)
    print(f"Loaded {len(X)} images, each of {X.shape[1]} pixels")

    n_components = min(n_components, X.shape[0])
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(X - np.mean(X, axis=0))

    embeddings = pca.transform(X - np.mean(X, axis=0))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    embedding_dict = {fname: emb for fname, emb in zip(filenames, embeddings)}
    print(f"PCA complete. Generated {embeddings.shape[1]}-D embeddings for {len(embeddings)} images.")
    return embedding_dict, pca


def compute_pca_embedding_std(embedding_dict):
    """Compute the mean vector, std vector, and average std deviation for PCA embeddings."""
    if len(embedding_dict) < 2:
        raise ValueError("Need at least 2 embeddings to compute standard deviation.")

    embeddings = np.stack(list(embedding_dict.values()))
    mean_vec = np.mean(embeddings, axis=0)
    std_vec = np.std(embeddings, axis=0)
    avg_std = np.mean(std_vec)

    print("\n📊 PCA Embedding Stats")
    print(f"Computed on {len(embeddings)} images")
    print(f"Average std deviation across PCA embedding dims: {avg_std:.5f}")
    print("First 10 elements of mean vector:", mean_vec[:10])
    print("First 10 elements of std vector:", std_vec[:10])
    print(f"Overall average std deviation: {avg_std:.5f}\n")

    return mean_vec, std_vec, avg_std


# =============================
# 🔹 MAIN EXECUTION
# =============================
if __name__ == "__main__":
    folder = "people/aditya"

    # --- PCA ---
    pca_embeddings, pca_model = compute_pca_embeddings(folder, n_components=128)
    pca_mean, pca_std, pca_avg_std = compute_pca_embedding_std(pca_embeddings)

    # --- DCT ---
    dct_mean, dct_std, dct_avg_std = compute_dct_folder_std(folder, embed_dim=128)

    # --- Compare both ---
    print("============== 📈 COMPARISON ==============")
    print(f"PCA  Avg Std Dev : {pca_avg_std:.5f}")
    print(f"DCT  Avg Std Dev : {dct_avg_std:.5f}")
    print(f"Ratio (PCA / DCT): {pca_avg_std / dct_avg_std:.2f}x higher variance\n")

    # --- Cosine similarity test (first two images) ---
    v1 = dct_embedding(os.path.join(folder, sorted(os.listdir(folder))[0]))
    v2 = dct_embedding(os.path.join(folder, sorted(os.listdir(folder))[1]))
    sim = cosine_similarity([v1], [v2])[0][0]
    print(f"Cosine similarity between first two DCT embeddings: {sim:.4f}")