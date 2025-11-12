# embeddings.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from scipy.fftpack import dct

# ================================================================
# 🔹 1. CNN EMBEDDING
# ================================================================

class CNNEmbedder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # reduces to (128,1,1)
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def cnn_embedding(img_path, model_path="face_embedder.pth", embedding_dim=128):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNNEmbedder(embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()
    return embedding


# ================================================================
# 🔹 2. DCT EMBEDDING
# ================================================================

def dct_embedding(img_path, embed_dim=128, size=(128, 128)):
    """
    Compute a purely mathematical embedding using 2D DCT.
    Returns a normalized numpy vector of dimension `embed_dim`.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    img = cv2.resize(img, size)

    # Apply 2D DCT
    dct_coeff = dct(dct(img.T, norm='ortho').T, norm='ortho')

    # Take low-frequency block
    block_size = int(np.ceil(np.sqrt(embed_dim)))
    dct_crop = dct_coeff[:block_size, :block_size].flatten()[:embed_dim]

    # Normalize
    dct_vec = dct_crop / np.linalg.norm(dct_crop)
    return dct_vec

