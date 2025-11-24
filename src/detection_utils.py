# # import cv2
# # import numpy as np
# # import torch
# # from torchvision import transforms
# # from PIL import Image
# # import mediapipe as mp

# # # load CNN and LSTM
# # from src.train_cnn import EyeCNN

# # from src.train_lstm import DrowsinessLSTM


# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # cnn_model = EyeCNN().to(device)
# # cnn_model.load_state_dict(torch.load("models/cnn_eye_model.pth", map_location=device))
# # cnn_model.eval()

# # lstm_model = DrowsinessLSTM().to(device)
# # lstm_model.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))
# # lstm_model.eval()

# # transform = transforms.Compose([
# #     transforms.Grayscale(),
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5,), (0.5,))
# # ])

# # mp_face = mp.solutions.face_mesh
# # LEFT = [33,160,158,133,153,144]
# # RIGHT = [362,385,387,263,373,380]

# # def crop_eyes(frame):
# #     h, w = frame.shape[:2]
# #     with mp_face.FaceMesh(
# #         static_image_mode=False, max_num_faces=1, refine_landmarks=True
# #     ) as fm:

# #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         res = fm.process(rgb)

# #         if not res.multi_face_landmarks:
# #             return None

# #         lm = res.multi_face_landmarks[0]

# #         pts = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in LEFT+RIGHT]
# #         pts = np.array(pts)

# #         x1,y1 = pts.min(axis=0)-5
# #         x2,y2 = pts.max(axis=0)+5

# #         x1, y1 = max(0,x1), max(0,y1)
# #         x2, y2 = min(w,x2), min(h,y2)

# #         crop = frame[y1:y2, x1:x2]
# #         if crop.size == 0:
# #             return None

# #         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# #         crop = cv2.resize(crop, (48,48))

# #         return crop


# # def cnn_predict_eye(crop):
# #     img = Image.fromarray(crop)
# #     x = transform(img).unsqueeze(0).to(device)

# #     with torch.no_grad():
# #         out = cnn_model(x)
# #         prob = torch.softmax(out, dim=1).cpu().numpy().squeeze()

# #     return prob    # [p_alert, p_drowsy]


# # def lstm_predict(sequence):
# #     x = torch.tensor([sequence], dtype=torch.float32).to(device)
# #     with torch.no_grad():
# #         out = lstm_model(x)
# #         cls = out.argmax(dim=1).item()
# #     return cls

# # src/detection_utils.py
# import cv2
# import numpy as np
# import torch
# from torchvision import transforms
# from PIL import Image
# import mediapipe as mp
# import os

# # --- Model definitions (only definitions; no training code) ---
# import torch.nn as nn

# class EyeCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64*12*12, 128), nn.ReLU(),
#             nn.Linear(128, 2)
#         )
#     def forward(self, x): return self.net(x)

# class DrowsinessLSTM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(2, 32, batch_first=True)
#         self.fc = nn.Linear(32, 2)
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# # --- device and model loading ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # instantiate models
# cnn_model = EyeCNN().to(device)
# lstm_model = DrowsinessLSTM().to(device)

# # load saved weights (make sure filenames match)
# cnn_path = os.path.join("models", "cnn_eye_model.pth")
# lstm_path = os.path.join("models", "lstm_model.pth")  # or lstm_model.pth / lstm_model.pt

# if os.path.exists(cnn_path):
#     cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
#     cnn_model.eval()
# else:
#     raise FileNotFoundError(f"CNN model not found: {cnn_path}")

# if os.path.exists(lstm_path):
#     lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
#     lstm_model.eval()
# else:
#     raise FileNotFoundError(f"LSTM model not found: {lstm_path}")

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# mp_face = mp.solutions.face_mesh
# LEFT = [33,160,158,133,153,144]
# RIGHT = [362,385,387,263,373,380]

# def crop_eyes(frame):
#     h, w = frame.shape[:2]
#     # Use a persistent FaceMesh instance for speed if you like (here we keep it simple)
#     with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = fm.process(rgb)
#         if not res.multi_face_landmarks:
#             return None
#         lm = res.multi_face_landmarks[0]
#         pts = [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in LEFT+RIGHT]
#         pts = np.array(pts)
#         x1,y1 = pts.min(axis=0)-5
#         x2,y2 = pts.max(axis=0)+5
#         x1, y1 = max(0,x1), max(0,y1)
#         x2, y2 = min(w,x2), min(h,y2)
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0: return None
#         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         crop = cv2.resize(crop, (48,48))
#         return crop

# def cnn_predict_eye(crop):
#     img = Image.fromarray(crop)
#     x = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = cnn_model(x)
#         prob = torch.softmax(out, dim=1).cpu().numpy().squeeze().tolist()
#     return prob   # [p_alert, p_drowsy]

# def lstm_predict(sequence):
#     x = torch.tensor([sequence], dtype=torch.float32).to(device)
#     with torch.no_grad():
#         out = lstm_model(x)
#         cls = int(out.argmax(dim=1).item())
#     return cls

# src/detection_utils.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import os

# Import model classes (make sure train files define classes only at top-level)
from src.train_cnn import EyeCNN
from src.train_lstm import DrowsinessLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
cnn_model = EyeCNN().to(device)
cnn_path = os.path.join("models", "cnn_eye_model.pth")
cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
cnn_model.eval()

lstm_model = DrowsinessLSTM().to(device)
lstm_path = os.path.join("models", "lstm_model.pth")
lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
lstm_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mp_face = mp.solutions.face_mesh
# These indices are commonly used, but you can tune if needed:
MOUTH_TOP = 13   # approximate top inner lip
MOUTH_BOTTOM = 14  # approximate bottom inner lip
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def crop_eyes(frame):
    """Return a 48x48 grayscale crop of both eyes area or None if no face."""
    h, w = frame.shape[:2]
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = []
        for i in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            pts.append((int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)))
        pts = np.array(pts)
        x1, y1 = pts.min(axis=0) - 8
        x2, y2 = pts.max(axis=0) + 8
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_resized = cv2.resize(crop_gray, (48, 48))
        return crop_resized

def compute_mar_from_landmarks(landmarks, img_w, img_h):
    """
    Compute a simple MAR = vertical_distance / horizontal_distance
    using MediaPipe landmark indices. Indices may be tuned if needed.
    """
    # convert normalized to pixel coords
    def xy(i):
        return np.array([landmarks[i].x * img_w, landmarks[i].y * img_h])

    try:
        top = xy(MOUTH_TOP)
        bottom = xy(MOUTH_BOTTOM)
        left = xy(MOUTH_LEFT)
        right = xy(MOUTH_RIGHT)

        vert = np.linalg.norm(top - bottom)
        horiz = np.linalg.norm(left - right) + 1e-6
        mar = vert / horiz
        return float(mar)
    except Exception:
        return 0.0

def cnn_predict_eye(crop):
    img = Image.fromarray(crop)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = cnn_model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy().squeeze()
    return prob  # [p_alert, p_drowsy]

def lstm_predict(sequence):
    # sequence: list of [p_alert, p_drowsy] (len T)
    x = torch.tensor([sequence], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = lstm_model(x)
        cls = out.argmax(dim=1).item()
    return cls

# Helper that does both: returns prob, mar (0 if not available)
def analyze_frame(frame):
    """
    Returns: (prob_array_or_None, mar_value)
    """
    # run face mesh once and compute both
    h, w = frame.shape[:2]
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None, 0.0
        lm = res.multi_face_landmarks[0].landmark
        mar = compute_mar_from_landmarks(lm, w, h)

    # get eye crop and probs (call crop_eyes which runs FaceMesh again - small overhead)
    eye_crop = crop_eyes(frame)
    if eye_crop is None:
        prob = np.array([1.0, 0.0])  # assume alert if eyes not detected
    else:
        prob = cnn_predict_eye(eye_crop)

    return prob, mar
