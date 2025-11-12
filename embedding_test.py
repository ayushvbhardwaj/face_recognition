import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ===== Define the same architecture you used during training =====
class FaceEmbedder(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = torch.nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

# ===== Define the test transform (no random augmentations) =====
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===== Function to compute embedding std deviation =====
def compute_embedding_std(model_path, image_folder, transform=test_transform, embedding_dim=128):
    """
    Loads a trained model (.pth), computes embeddings for all images in a folder,
    and returns the mean vector, std vector, and average std value.
    """
    # Choose device (Apple GPU if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = FaceEmbedder(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Collect embeddings
    embeddings = []
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(image_folder, fname)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).cpu().numpy()[0]
        embeddings.append(emb)

    embeddings = np.stack(embeddings)
    mean_vec = np.mean(embeddings, axis=0)
    std_vec = np.std(embeddings, axis=0)
    avg_std = np.mean(std_vec)

    print(f"Computed on {len(embeddings)} images")
    print(f"Average std deviation across embedding dims: {avg_std:.5f}")

    return mean_vec, std_vec, avg_std


model_path = "face_embedder.pth"
image_folder = "people/aditya"

mean_vec, std_vec, avg_std = compute_embedding_std(model_path, image_folder)

print("\nFirst 10 elements of mean vector:", mean_vec[:10])
print("First 10 elements of std vector:", std_vec[:10])
print(f"\nOverall average std: {avg_std:.5f}")