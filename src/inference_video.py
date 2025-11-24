# import cv2
# import numpy as np
# from .detection_utils import crop_eyes, cnn_predict_eye, lstm_predict

# def process_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     out = cv2.VideoWriter(
#         output_path,
#         cv2.VideoWriter_fourcc(*"XVID"),
#         fps,
#         (w, h)
#     )

#     T = 15
#     seq = []
#     timeline = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         crop = crop_eyes(frame)
#         if crop is not None:
#             prob = cnn_predict_eye(crop)
#             seq.append(prob)
#         else:
#             seq.append([1, 0])  # assume alert if no face

#         if len(seq) > T:
#             seq.pop(0)

#         if len(seq) == T:
#             cls = lstm_predict(seq)
#             state = "DROWSY" if cls == 1 else "ALERT"
#         else:
#             state = "ALERT"

#         timeline.append(state)

#         # draw on frame
#         color = (0,0,255) if state=="DROWSY" else (0,255,0)
#         cv2.putText(frame, state, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#         out.write(frame)

#     cap.release()
#     out.release()

#     return timeline

# src/inference_video.py
import cv2
import numpy as np
import os
from src.detection_utils import analyze_frame, lstm_predict

def process_video(input_path, output_path_mp4, T=15, mar_thresh=0.6, mar_consec=3):
    """
    Processes video and writes MP4 (mp4v codec) to output_path_mp4.
    Returns: (timeline_list, output_path_mp4)
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # ensure folder exists
    os.makedirs(os.path.dirname(output_path_mp4) or ".", exist_ok=True)

    # mp4v codec works without ffmpeg in most environments
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (w, h))

    seq = []
    timeline = []
    mar_window = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prob, mar = analyze_frame(frame)  # prob = [p_alert, p_drowsy], mar = float
        seq.append(prob)
        if len(seq) > T:
            seq.pop(0)

        # LSTM decision when we have a full sequence
        lstm_drowsy = False
        if len(seq) == T:
            cls = lstm_predict(seq)
            lstm_drowsy = (cls == 1)

        # MAR yawning rule
        if mar > mar_thresh:
            mar_window += 1
        else:
            mar_window = 0
        mar_drowsy = (mar_window >= mar_consec)

        # final decision: drowsy if LSTM OR MAR detected
        state = "DROWSY" if (lstm_drowsy or mar_drowsy) else "ALERT"
        timeline.append(state)

        # draw on frame
        color = (0, 0, 255) if state == "DROWSY" else (0, 255, 0)
        cv2.putText(frame, state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        # optionally show MAR value
        cv2.putText(frame, f"MAR:{mar:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return timeline, output_path_mp4
