⸻

Face Recognition Ensemble

This project implements a modular face recognition system that uses multiple types of feature embeddings — a CNN trained on the VGGFace2 dataset, a DCT-based descriptor, and a pretrained FaceNet model — for robust identity recognition.

The system supports embedding generation, dataset management, live recognition through a webcam, and 3D visualization of feature vectors.

⸻

Overview

The repository contains:
	•	Tools to generate and store averaged embeddings for each person.
	•	Scripts for real-time recognition using MediaPipe face detection.
	•	A visualization module to project high-dimensional embeddings into 3D space.
	•	A CNN trained on VGGFace2 for extracting 128D face embeddings.
	•	DCT and FaceNet alternatives for experimentation and comparison.

⸻

Directory Structure

face_rec/
│
├── cnn_embed.py             # CNN embedding model (trained on VGGFace2)
├── create_db.py             # Utility to create person folders and manage data
├── dataset_embedder.py      # Averages embeddings for each person and saves JSON
├── embedding_test.py        # Testing different embeddings for a single image
├── embeddings.py            # Contains cnn_embedding() and dct_embedding() functions
├── face_embedder.pth        # Saved weights for the CNN trained on VGGFace2
├── main.py                  # Real-time face recognition using MediaPipe
├── pca_embed.py             # PCA-based dimensionality reduction experiments
├── visualize.py             # Visualizes embeddings in 3D using PCA
│
├── people/                  # Directory containing folders for each person
│   ├── ayush/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── embeddings.json
│   └── aditya/
│       └── embeddings.json
│
└── train/                   # Training utilities and datasets


⸻

Embeddings

Three types of embeddings are used:

Type	Dimension	Description
CNN	128	Custom CNN trained on VGGFace2; lightweight and fast
DCT	128	Frequency-domain descriptor using Discrete Cosine Transform
FaceNet	512	Pretrained InceptionResNetV1 model for identity features

Each person’s embedding file (embeddings.json) stores the averaged embedding for each type.

Example:

{
  "average_cnn": [...],
  "average_dct": [...],
  "average_facenet": [...]
}


⸻

Usage

1. Generate averaged embeddings

Place your images inside a folder under people/, for example:

people/ayush/img1.jpg
people/ayush/img2.jpg

Then run:

python dataset_embedder.py people/ayush

This creates people/ayush/embeddings.json containing averaged embeddings.

⸻

2. Run live recognition

To start real-time face recognition using your webcam:

python main.py

You can choose between:
	•	Using a single model (CNN, DCT, or FaceNet)
	•	Or enabling all three models for combined recognition.

The script detects faces using MediaPipe and compares the current face embedding with stored averages using cosine similarity.

⸻

3. Visualize embeddings

To view 3D projections of embeddings:

python visualize.py

This reduces each embedding to 3D space using PCA and plots the resulting vectors for comparison across models.

⸻

Notes
	•	The CNN model (face_embedder.pth) was trained on the VGGFace2 dataset.
	•	FaceNet uses the pretrained InceptionResNetV1 model from facenet-pytorch.
	•	The DCT embedding method is purely frequency-based and does not require training.

⸻

Requirements
	•	Python 3.8+
	•	PyTorch
	•	OpenCV
	•	MediaPipe
	•	NumPy
	•	scikit-learn
	•	Matplotlib
	•	tqdm

Install dependencies:

pip install -r requirements.txt


⸻

Performance

Approximate CPU performance on a laptop or desktop:

Model	Embedding Dim	Accuracy	Speed
CNN	128	~90%	20–40 FPS
DCT	128	~70%	40–60 FPS
FaceNet	512	~95%	10–15 FPS
Ensemble (All)	—	~95–97%	10–20 FPS


⸻

Acknowledgements
	•	VGGFace2 Dataset – for training the CNN model.
	•	FaceNet (InceptionResNetV1) – pretrained model from facenet-pytorch.
	•	MediaPipe – used for real-time face detection.

⸻
