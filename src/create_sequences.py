import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import csv

# Load trained CNN
from train_cnn import EyeCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EyeCNN().to(device)
model.load_state_dict(torch.load("models/cnn_eye_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

crop_root = "data/crops"
label_root = "data/labels"

all_feats = []
all_labels = []

T = 15   # sequence length

for vid_folder in os.listdir(crop_root):
    crop_dir = os.path.join(crop_root, vid_folder)
    label_csv = os.path.join(label_root, f"{vid_folder}_labels.csv")

    if not os.path.exists(label_csv):
        continue

    # Load labels
    label_map = {}
    with open(label_csv) as f:
        for fname, lbl in csv.reader(f):
            label_map[fname] = lbl

    files = sorted(label_map.keys())

    video_features = []
    video_labels = []

    # CNN predictions for each frame
    for fname in files:
        img_path = os.path.join(crop_dir, fname)
        img = Image.open(img_path).convert("L")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy().squeeze()

        video_features.append(prob)

        label = 0 if label_map[fname] == "alert" else 1
        video_labels.append(label)

    # Convert to sequences
    for i in range(len(video_features) - T + 1):
        seq_feat = video_features[i:i+T]
        seq_label = max(video_labels[i:i+T], key=video_labels[i:i+T].count)

        all_feats.append(seq_feat)
        all_labels.append(seq_label)

np.save("data/sequences.npy", np.array(all_feats))
np.save("data/seq_labels.npy", np.array(all_labels))

print("Saved sequences:", len(all_feats))
