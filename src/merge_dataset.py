import os, csv, shutil, random

crop_root = "data/crops"
label_root = "data/labels"
dest = "dataset"
os.makedirs(dest, exist_ok=True)

train_alert = os.path.join(dest, "train/alert");   os.makedirs(train_alert, exist_ok=True)
train_drowsy = os.path.join(dest, "train/drowsy"); os.makedirs(train_drowsy, exist_ok=True)
val_alert = os.path.join(dest, "val/alert");       os.makedirs(val_alert, exist_ok=True)
val_drowsy = os.path.join(dest, "val/drowsy");     os.makedirs(val_drowsy, exist_ok=True)

samples = []

for vid_folder in os.listdir(crop_root):
    crop_folder = os.path.join(crop_root, vid_folder)
    label_file = os.path.join(label_root, f"{vid_folder}_labels.csv")

    if not os.path.exists(label_file):
        continue

    with open(label_file, "r") as f:
        for fname, lbl in csv.reader(f):
            samples.append((os.path.join(crop_folder, fname), lbl))

random.shuffle(samples)
split = int(0.8*len(samples))

for i,(path, lbl) in enumerate(samples):
    fname = os.path.basename(path)
    if i < split:
        dst = train_alert if lbl=="alert" else train_drowsy
    else:
        dst = val_alert if lbl=="alert" else val_drowsy
    shutil.copy(path, os.path.join(dst, fname))

print(f"[DONE] FINAL DATASET CREATED WITH {len(samples)} SAMPLES")
