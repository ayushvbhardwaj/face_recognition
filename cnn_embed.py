import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)



class FaceEmbedder(nn.Module):
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
            nn.AdaptiveAvgPool2d((1,1)) 
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x


class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = {cls: os.listdir(os.path.join(root_dir, cls)) for cls in self.classes}

    def __len__(self):
        return sum(len(v) for v in self.images.values())

    def __getitem__(self, idx):
        cls_anchor = random.choice(self.classes)
        pos_images = self.images[cls_anchor]
        if len(pos_images) < 2:
            return self.__getitem__(idx)  # skip if not enough samples

        anchor_path, pos_path = random.sample(pos_images, 2)
        anchor = Image.open(os.path.join(self.root_dir, cls_anchor, anchor_path)).convert("RGB")
        positive = Image.open(os.path.join(self.root_dir, cls_anchor, pos_path)).convert("RGB")

        # choose negative
        neg_cls = random.choice([c for c in self.classes if c != cls_anchor])
        neg_img = random.choice(self.images[neg_cls])
        negative = Image.open(os.path.join(self.root_dir, neg_cls, neg_img)).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


# transformations for training images
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

if __name__ == "__main__":
    dataset = TripletFaceDataset('train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = FaceEmbedder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.TripletMarginLoss(margin=1.0)

    print("Starting training...")

    for epoch in range(20):
        print(f"Starting Epoch {epoch+1}/20")

        start_time = time.time()

        model.train()
        total_loss = 0
        for anchor, pos, neg in loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)
            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch [{epoch+1}/20], Loss: {total_loss/len(loader):.4f}, Time: {epoch_time:.2f} sec")

    torch.save(model.state_dict(), 'face_embedder.pth')
    print("Model saved.")
