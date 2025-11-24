import os

video_dir = "videos"

files = [f for f in os.listdir(video_dir) if f.lower().endswith(".avi")]
files.sort()

for i, f in enumerate(files, start=1):
    new_name = f"video{i}.avi"
    old_path = os.path.join(video_dir, f)
    new_path = os.path.join(video_dir, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed {f} -> {new_name}")

print("All videos renamed successfully.")
