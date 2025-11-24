import cv2, os

video_dir = "videos"
frame_dir = "data/frames"
os.makedirs(frame_dir, exist_ok=True)

for vid in os.listdir(video_dir):
    if not vid.lower().endswith(".avi"):
        continue
    
    video_path = os.path.join(video_dir, vid)
    video_name = vid.split(".")[0]
    save_dir = os.path.join(frame_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(save_dir, f"{count:06d}.jpg"), frame)
        count += 1

    cap.release()
    print(f"[DONE] Extracted {count} frames from {vid}")
