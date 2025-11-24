import cv2
import os
import csv
import math

crop_dir = "data/crops"
label_dir = "data/labels"
os.makedirs(label_dir, exist_ok=True)

def scale_to_window(img, max_w=1200, max_h=800):
    h,w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        new_w = math.floor(w * scale)
        new_h = math.floor(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

videos = sorted([d for d in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, d))])

if not videos:
    print("No video folders found in", crop_dir)
    raise SystemExit

print("Found video folders:", videos)
print("Controls: A = alert, D = drowsy, S = skip, B = back, Q = quit/save and next video")
print("Click on the image window to ensure it has focus before pressing keys.\n")

for vid in videos:
    print(f"\n=== Labelling video: {vid} ===")
    folder = os.path.join(crop_dir, vid)
    out_csv = os.path.join(label_dir, f"{vid}_labels.csv")

    # load existing labels if present (so you can resume)
    labels = {}
    if os.path.exists(out_csv):
        try:
            with open(out_csv, "r", newline="") as f:
                r = csv.reader(f)
                for row in r:
                    if row:
                        labels[row[0]] = row[1]
            print(f"Loaded existing labels ({len(labels)}) from {out_csv}")
        except Exception as e:
            print("Warning: failed to read existing CSV:", e)

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not files:
        print("No image files in", folder, "- skipping.")
        continue

    i = 0
    # If there are already labels, start from the first unlabeled index
    while i < len(files) and files[i] in labels:
        i += 1

    window_name = f"Label {vid} (A=alert, D=drowsy, S=skip, B=back, Q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while i < len(files):
        fname = files[i]
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print("Warning: failed to read", path, "- skipping")
            i += 1
            continue

        disp = scale_to_window(img)
        # overlay text
        txt = f"{i+1}/{len(files)}  {fname}"
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        if fname in labels:
            cv2.putText(disp, f"LABEL: {labels[fname]}", (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(0) & 0xFF

        # map keys (lowercase/uppercase)
        if key in (ord('a'), ord('A')):
            labels[fname] = "alert"
            i += 1
        elif key in (ord('d'), ord('D')):
            labels[fname] = "drowsy"
            i += 1
        elif key in (ord('s'), ord('S')):
            # skip without labeling
            i += 1
        elif key in (ord('b'), ord('B')):
            i = max(0, i-1)
        elif key in (ord('q'), ord('Q')):
            print("Quitting labeling for this video and saving progress...")
            break
        else:
            print("Unrecognized key (press A/D/S/B/Q).")

        # save progress after each change
        try:
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                for k, v in labels.items():
                    w.writerow([k, v])
        except Exception as e:
            print("Error saving labels:", e)

    cv2.destroyWindow(window_name)
    # final save for video
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        for k, v in labels.items():
            w.writerow([k, v])
    print(f"[SAVED] {out_csv}  (total labeled: {len(labels)})")

print("\nLabeling complete for all video folders.")
cv2.destroyAllWindows()
