#!/usr/bin/env python3
"""
Evaluate CNN (frame-wise) and LSTM (sequence-wise) predictions on labeled videos.

Usage:
    python src/evaluate.py --limit 20 --crops data/crops --labels data/labels

Outputs:
 - Terminal-friendly evaluation summary
 - JSON file: results_evaluation.json (project root)
"""
import os
import sys
import time
import argparse
import json
from glob import glob

# ensure project root on sys.path so `src` imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm

# optional pretty table
try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except Exception:
    _HAS_TABULATE = False

# sklearn metrics
try:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e

# Import your inference utilities (must NOT retrain)
try:
    from src.detection_utils import crop_eyes, cnn_predict_eye, lstm_predict
except Exception as e:
    raise RuntimeError(
        "Failed to import src.detection_utils. Ensure src/ is a package (src/__init__.py) "
        "and files exist. Error: " + str(e)
    ) from e

SEQ_LEN = 15  # must match training sequence length


def load_label_map_for_video(label_dir, video_folder):
    csv_path = os.path.join(label_dir, f"{video_folder}_labels.csv")
    if not os.path.exists(csv_path):
        return None
    label_map = {}
    with open(csv_path, "r", newline="") as f:
        for line in f:
            row = line.strip().split(",")
            if len(row) >= 2:
                fname, lbl = row[0].strip(), row[1].strip()
                # 0 = alert, 1 = drowsy
                label_map[fname] = 0 if lbl.lower().startswith("alert") else 1
    return label_map


def pretty_confusion(cm):
    """Return (tn, fp, fn, tp) safely even if cm shape is unexpected."""
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # fewer classes or unexpected shape: zero-fill
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(gts, preds, times):
    """
    Compute precision/recall/f1, confusion matrix, false alarm rate, and inference fps.
    Expects binary labels (0 negative/alert, 1 positive/drowsy).
    """
    if len(gts) == 0:
        return None

    p, r, f1, _ = precision_recall_fscore_support(gts, preds, average="binary", pos_label=1)
    cm = confusion_matrix(gts, preds)
    tn, fp, fn, tp = pretty_confusion(cm)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    inf_fps = len(times) / sum(times) if sum(times) > 0 else 0.0

    return {
        "n": int(len(gts)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_alarm_rate": float(false_alarm_rate),
        "inference_fps": float(inf_fps)
    }


def evaluate_on_videos(crops_root="data/crops", labels_root="data/labels", videos_limit=20):
    video_folders = sorted(
        [d for d in os.listdir(crops_root) if os.path.isdir(os.path.join(crops_root, d))]
    )
    if videos_limit:
        video_folders = video_folders[:videos_limit]

    all_frame_gts = []
    all_frame_preds = []
    all_frame_times = []

    all_seq_gts = []
    all_seq_preds = []
    all_seq_times = []

    processed_videos = 0
    missing_label_count = 0

    pbar = tqdm(video_folders, desc="Videos", unit="vid")
    for vid in pbar:
        crop_folder = os.path.join(crops_root, vid)
        label_map = load_label_map_for_video(labels_root, vid)
        if label_map is None:
            missing_label_count += 1
            pbar.set_postfix_str(f"skip:{vid} (no labels)")
            continue

        # Only evaluate frames that have ground-truth labels
        frame_files = sorted([f for f in os.listdir(crop_folder) if f in label_map])
        if not frame_files:
            # no labeled frames in this crop folder
            continue

        # Per-frame CNN predictions
        cnn_probs = []
        cnn_frame_times = []
        gts = []
        for fname in frame_files:
            path = os.path.join(crop_folder, fname)
            # load grayscale crop
            try:
                import cv2
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
            except Exception:
                from PIL import Image
                img = np.array(Image.open(path).convert("L"))

            t0 = time.time()
            prob = cnn_predict_eye(img)  # expected: [p_alert, p_drowsy]
            t1 = time.time()

            cnn_probs.append(prob)
            cnn_frame_times.append(t1 - t0)
            gts.append(label_map[fname])

        if len(cnn_probs) == 0:
            continue

        cnn_preds = [int(np.argmax(p)) for p in cnn_probs]
        all_frame_gts.extend(gts)
        all_frame_preds.extend(cnn_preds)
        all_frame_times.extend(cnn_frame_times)

        # prepare sequence windows for LSTM
        L = len(cnn_probs)
        seq_feats = []
        seq_gts = []
        for i in range(0, L - SEQ_LEN + 1):
            window_probs = cnn_probs[i:i + SEQ_LEN]
            seq_feats.append(window_probs)
            window_labels = gts[i:i + SEQ_LEN]
            seq_gts.append(1 if sum(window_labels) > (SEQ_LEN / 2) else 0)

        # run LSTM predictions for each sequence and time them
        for seq in seq_feats:
            t0 = time.time()
            cls = lstm_predict(seq)  # expected 0 or 1
            t1 = time.time()
            all_seq_preds.append(int(cls))
            all_seq_times.append(t1 - t0)

        all_seq_gts.extend(seq_gts)
        processed_videos += 1
        pbar.set_postfix_str(f"processed:{processed_videos}")

    # compute metrics
    frame_metrics = compute_metrics(all_frame_gts, all_frame_preds, all_frame_times)
    seq_metrics = compute_metrics(all_seq_gts, all_seq_preds, all_seq_times) if len(all_seq_gts) > 0 else None

    results = {
        "processed_videos": int(processed_videos),
        "missing_label_files": int(missing_label_count),
        "frame_level": frame_metrics,
        "sequence_level": seq_metrics
    }

    # save json
    out_json = os.path.join(ROOT, "results_evaluation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_results(results):
    print("\n=== EVALUATION SUMMARY ===\n")
    print(f"Processed videos: {results.get('processed_videos')}")
    print(f"Missing label files: {results.get('missing_label_files')}\n")

    # Frame-level (CNN)
    fl = results.get("frame_level")
    if fl:
        print("== Frame-level (CNN) ==")
        rows = [
            ["Total frames", fl["n"]],
            ["Precision (drowsy)", f"{fl['precision']:.4f}"],
            ["Recall (drowsy)", f"{fl['recall']:.4f}"],
            ["F1 (drowsy)", f"{fl['f1']:.4f}"],
            ["TN", fl["tn"]],
            ["FP", fl["fp"]],
            ["FN", fl["fn"]],
            ["TP", fl["tp"]],
            ["False alarm rate", f"{fl['false_alarm_rate']:.4f}"],
            ["Inference FPS (per frame, CNN)", f"{fl['inference_fps']:.2f}"]
        ]
        if _HAS_TABULATE:
            print(tabulate(rows, tablefmt="pretty"))
        else:
            for k, v in rows:
                print(f"{k:30s} {v}")
        print("\n")

    # Sequence-level (LSTM)
    sl = results.get("sequence_level")
    if sl:
        print("== Sequence-level (LSTM) ==")
        rows = [
            ["Total sequences", sl["n"]],
            ["Precision (drowsy)", f"{sl['precision']:.4f}"],
            ["Recall (drowsy)", f"{sl['recall']:.4f}"],
            ["F1 (drowsy)", f"{sl['f1']:.4f}"],
            ["TN", sl["tn"]],
            ["FP", sl["fp"]],
            ["FN", sl["fn"]],
            ["TP", sl["tp"]],
            ["False alarm rate", f"{sl['false_alarm_rate']:.4f}"],
            ["Inference FPS (per sequence, LSTM)", f"{sl['inference_fps']:.2f}"]
        ]
        if _HAS_TABULATE:
            print(tabulate(rows, tablefmt="pretty"))
        else:
            for k, v in rows:
                print(f"{k:30s} {v}")
        print("\n")

    print(f"Results were also saved to: {os.path.join(ROOT, 'results_evaluation.json')}\n")

    # Explanation: why both metrics
    print("Note: 'frame-level' metrics evaluate the CNN's performance on single frames (open vs closed).")
    print("'sequence-level' metrics evaluate the LSTM's performance on sliding windows (temporal drowsiness detection).")
    print("Both are useful: CNN shows frame-wise accuracy; LSTM shows how well temporal patterns are detected.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default="data/crops", help="Root folder with crop folders (per video)")
    parser.add_argument("--labels", default="data/labels", help="Root folder with labels (per video CSV)")
    parser.add_argument("--limit", type=int, default=20, help="Max number of videos to evaluate")
    args = parser.parse_args()

    print("Evaluating with:")
    print("  crops:", args.crops)
    print("  labels:", args.labels)
    print("  limit:", args.limit)
    results = evaluate_on_videos(crops_root=args.crops, labels_root=args.labels, videos_limit=args.limit)
    print_results(results)


if __name__ == "__main__":
    main()
