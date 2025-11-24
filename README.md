# 📘 **Driver Drowsiness Detection**

This project detects **driver drowsiness** using:

* **CNN (frame-level eye state detection)**
* **LSTM (sequence-level drowsiness detection)**
* **Face landmark features (EAR / MAR)** *(optional for evaluation)*
* A **Streamlit dashboard** for uploading videos and visualizing predictions.

The system supports:

✔ Video input (MP4/AVI)
✔ CNN + LSTM inference
✔ Frame-wise and sequence-wise evaluation
✔ Metrics: Precision, Recall, F1-Score, False Alarm Rate, Inference FPS
✔ JSON report generation
✔ Streamlit UI for video output + charts

---

## 📂 **Project Structure**

```
driver_drowsy/
│── app.py                      # Streamlit dashboard
│── results_evaluation.json     # Saved evaluation results
│── models/
│     ├── cnn_eye_model.pth
│     └── lstm_model.pth
│── data/
│     ├── crops/                # Cropped eye frames per video
│     └── labels/               # CSV labels per video
│── src/
│     ├── detection_utils.py
│     ├── inference_video.py
│     ├── train_cnn.py
│     ├── train_lstm.py
│     └── evaluate.py
└── README.md
```

---

# 🛠 **Installation**

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install streamlit opencv-python numpy pillow torch torchvision plotly scikit-learn tqdm mediapipe
```

---

# ▶️ **Running the Streamlit App**

```bash
streamlit run app.py
```

This opens a browser window where you can:

* Upload a video
* View processed predictions
* See summary metrics & timeline plot
* Download processed output video

---

# 📊 **Evaluating the Model**

Run evaluation on up to 20 labeled videos:

```bash
python src/evaluate.py --limit 20
```

This generates:

* **Terminal output**
* **results_evaluation.json**

Metrics include:

* Precision
* Recall
* F1 score
* False Alarm Rate
* Inference FPS
* Confusion matrix

---

# 📁 **Training (Optional)**

If you want to retrain:

```bash
python src/train_cnn.py
python src/create_sequences.py
python src/train_lstm.py
```

Models will be saved in:

```
models/
```

---

# 🎬 **Inference Script**

To process a single video manually:

```bash
from src.inference_video import process_video

timeline, mp4_path = process_video("input.mp4", "output.mp4")
```


