import cv2, os
import numpy as np
import mediapipe as mp

frame_dir = "data/frames"
crop_dir = "data/crops"
os.makedirs(crop_dir, exist_ok=True)

mp_face = mp.solutions.face_mesh
LEFT = [33,160,158,133,153,144]
RIGHT = [362,385,387,263,373,380]

with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:

    for folder in os.listdir(frame_dir):
        src_folder = os.path.join(frame_dir, folder)
        dst_folder = os.path.join(crop_dir, folder)
        os.makedirs(dst_folder, exist_ok=True)

        for img_name in sorted(os.listdir(src_folder)):
            img_path = os.path.join(src_folder, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            h,w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0]
            points = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in LEFT+RIGHT]
            pts = np.array(points)

            x1,y1 = pts.min(axis=0)-8
            x2,y2 = pts.max(axis=0)+8
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue

            eye = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            eye = cv2.resize(eye, (48,48))

            cv2.imwrite(os.path.join(dst_folder, img_name), eye)

        print(f"[DONE] Cropped eyes from video: {folder}")
