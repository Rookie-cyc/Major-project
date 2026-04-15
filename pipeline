# ==========================================================
# AI COLLISION PREDICTION SYSTEM
# YOLOv12 (best.pt) + MiDaS Small + GRU + TTC
# ==========================================================

import os
import cv2
import torch
import numpy as np
from collections import deque
import torch.nn as nn

# ==========================================================
# CONFIG
# ==========================================================
VIDEO_SOURCE = 0   # 0 = webcam OR "input.mp4"
YOLO_MODEL_PATH = "best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQUENCE_LENGTH = 10
WARNING_THRESHOLD = 0.35
DANGER_THRESHOLD = 0.55

# ==========================================================
# LOAD YOLOv12 (best.pt)
# ==========================================================
from ultralytics import YOLO

print("Loading YOLOv12 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# ==========================================================
# LOAD MiDaS SMALL
# ==========================================================
print("Loading MiDaS model...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(DEVICE)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
depth_transform = transforms.small_transform

# ==========================================================
# GRU MODEL
# ==========================================================
class GRUModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Load GRU (replace with trained model if available)
gru_model = GRUModel().to(DEVICE)
gru_model.eval()

print("All models loaded successfully!")

# ==========================================================
# DEPTH FUNCTION
# ==========================================================
def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = depth_transform(img).to(DEVICE)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

# ==========================================================
# TTC COMPUTATION
# ==========================================================
def compute_ttc(prev_depth, curr_depth, epsilon=1e-6):
    if prev_depth is None:
        return 0.0

    prev_mean = np.mean(prev_depth)
    curr_mean = np.mean(curr_depth)

    ttc = prev_mean / (prev_mean - curr_mean + epsilon)
    return np.clip(ttc, 0, 10)

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(detections, depth_map, ttc):
    num_objects = len(detections.boxes) if detections.boxes is not None else 0
    avg_depth = float(np.mean(depth_map))

    return np.array([num_objects, avg_depth, ttc], dtype=np.float32)

# ==========================================================
# ALERT LOGIC
# ==========================================================
def get_alert(risk):
    if risk >= DANGER_THRESHOLD:
        return "DANGER"
    elif risk >= WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "SAFE"

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_pipeline():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prev_depth = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------- YOLO DETECTION --------
        results = yolo_model(frame)[0]

        # -------- DEPTH --------
        depth_map = estimate_depth(frame)

        # -------- TTC --------
        ttc = compute_ttc(prev_depth, depth_map)

        # -------- FEATURES --------
        features = extract_features(results, depth_map, ttc)
        feature_buffer.append(features)

        # -------- GRU PREDICTION --------
        if len(feature_buffer) == SEQUENCE_LENGTH:
            seq = np.array(feature_buffer)
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                risk = float(gru_model(seq).item())
        else:
            risk = 0.0

        alert = get_alert(risk)

        # -------- VISUALIZATION --------
        annotated = results.plot()

        # Risk text
        cv2.putText(annotated, f"Risk: {risk:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Alert text
        color = (0, 255, 0)
        if alert == "WARNING":
            color = (0, 255, 255)
        elif alert == "DANGER":
            color = (0, 0, 255)

        cv2.putText(annotated, f"Alert: {alert}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # TTC display
        cv2.putText(annotated, f"TTC: {ttc:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Collision Prediction System", annotated)

        prev_depth = depth_map

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    run_pipeline()
