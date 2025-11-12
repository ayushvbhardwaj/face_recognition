# Face Recognition Ensemble

This project implements a modular face recognition system that uses multiple types of feature embeddings — a CNN trained on the **VGGFace2** dataset, a **DCT-based descriptor**, and a pretrained **FaceNet** model — for robust identity recognition.

The system supports embedding generation, dataset management, live recognition through a webcam, and 3D visualization of feature vectors.

---

## Overview

This repository provides:

- Tools to generate and store averaged embeddings for each person  
- Real-time face recognition using **MediaPipe** face detection  
- A visualization module to project high-dimensional embeddings into 3D space  
- A CNN model trained on **VGGFace2** for 128D embeddings  
- DCT and FaceNet alternatives for experimentation and comparison  

---

## Directory Structure
face_rec/
│
├── cnn_embed.py             # CNN embedding model (trained on VGGFace2)
├── create_db.py             # Utility to create person folders and manage data
├── dataset_embedder.py      # Computes averaged embeddings for each person
├── embedding_test.py        # Tests embedding generation on a single image
├── embeddings.py            # Contains cnn_embedding() and dct_embedding() functions
├── face_embedder.pth        # Trained CNN weights on VGGFace2
├── main.py                  # Real-time webcam recognition using MediaPipe
├── pca_embed.py             # PCA-based embedding compression experiments
├── visualize.py             # 3D PCA visualization of average embeddings
│
├── people/                  # Each person’s folder with images + embeddings.json
│   ├── ayush/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── embeddings.json
│   ├── aditya/
│   │   ├── img1.jpg
│   │   └── embeddings.json
│
└── train/                   # Training utilities and datasets


---

## Embeddings

Three types of embeddings are supported:

| Type | Dimension | Description |
|------|------------|-------------|
| CNN | 128 | Custom CNN trained on VGGFace2; fast and compact |
| DCT | 128 | Frequency-domain representation (Discrete Cosine Transform) |
| FaceNet | 512 | Pretrained InceptionResNetV1 model from facenet-pytorch |

Each person’s averaged embedding is stored as `embeddings.json` in their folder:

```json
{
  "average_cnn": [...],
  "average_dct": [...],
  "average_facenet": [...]
}
```

Usage

1. Generate averaged embeddings

Organize images for each person in the people/ folder:

people/
└── ayush/
    ├── img1.jpg
    ├── img2.jpg

Then run:

python dataset_embedder.py people/ayush

This creates:

people/ayush/embeddings.json

2. Run real-time recognition

To recognize faces using your webcam:

python main.py

You can configure:
	•	Which model to use (CNN, DCT, or FACENET)
	•	Or run in combined ensemble mode using all three

The script detects faces via MediaPipe, extracts embeddings, and compares them with stored averages using cosine similarity.

Requirements
	•	Python 3.8+
	•	PyTorch
	•	OpenCV
	•	MediaPipe
	•	NumPy
	•	scikit-learn
	•	Matplotlib
	•	tqdm

Install all dependencies:

Notes
	•	The CNN model (face_embedder.pth) is trained on the VGGFace2 dataset.
	•	FaceNet uses the pretrained InceptionResNetV1 model from facenet-pytorch.
	•	The DCT embedding method is frequency-based and requires no training.


	---

✅ This version:
- Displays properly on GitHub  
- Preserves indentation and alignment in directory trees  
- Keeps comments readable and inline  
- Avoids emojis and unnecessary formatting  
- Uses triple backticks for every code block  

Would you like me to include a **short results section** at the end (showing example terminal outputs and what the 3D plot looks like)? It’d make your repo presentation feel complete.
