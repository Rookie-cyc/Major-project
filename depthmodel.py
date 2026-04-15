# ===========================
# Depth Estimation on KITTI
# ===========================

import os
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ===========================
# CONFIG
# ===========================
KITTI_ROOT = "data/kitti"   # change this to your dataset path
KITTI_VAL_RGB = os.path.join(KITTI_ROOT, "val")
KITTI_GT_DEPTH = os.path.join(KITTI_ROOT, "data_depth_selection")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BENCHMARK_N = 100
INPUT_SIZE = (640, 192)

# ===========================
# LOAD MiDaS MODEL
# ===========================
print("Loading MiDaS model...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

print("Model loaded!")

# ===========================
# FIND IMAGE PAIRS
# ===========================
def find_kitti_pairs(rgb_root, depth_root, n=100):
    pairs = []

    rgb_files = (
        glob.glob(os.path.join(rgb_root, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(rgb_root, "**", "*.jpg"), recursive=True)
    )

    depth_files = glob.glob(os.path.join(depth_root, "**", "*.png"), recursive=True)

    depth_lookup = {os.path.basename(dp): dp for dp in depth_files}

    for rp in rgb_files:
        fname = os.path.basename(rp)
        if fname in depth_lookup:
            pairs.append((rp, depth_lookup[fname]))
        if len(pairs) >= n:
            break

    print(f"Matched {len(pairs)} pairs")
    return pairs


# ===========================
# DEPTH PREDICTION
# ===========================
def predict_depth(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return depth


# ===========================
# METRICS
# ===========================
def compute_metrics(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    mae = np.mean(np.abs(gt - pred))

    # R²
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return rmse, mae, r2


# ===========================
# LOAD GT DEPTH
# ===========================
def load_gt_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 256.0
    return depth


# ===========================
# MAIN BENCHMARK LOOP
# ===========================
pairs = find_kitti_pairs(KITTI_VAL_RGB, KITTI_GT_DEPTH, BENCHMARK_N)

rmse_list, mae_list, r2_list = [], [], []

print("Running benchmark...")

for rgb_path, depth_path in tqdm(pairs):
    pred_depth = predict_depth(rgb_path)
    gt_depth = load_gt_depth(depth_path)

    # Resize GT to match prediction
    gt_depth = cv2.resize(gt_depth, (pred_depth.shape[1], pred_depth.shape[0]))

    rmse, mae, r2 = compute_metrics(pred_depth, gt_depth)

    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)

# ===========================
# RESULTS
# ===========================
print("\n===== FINAL RESULTS =====")
print(f"RMSE: {np.mean(rmse_list):.4f}")
print(f"MAE : {np.mean(mae_list):.4f}")
print(f"R²  : {np.mean(r2_list):.4f}")

# ===========================
# SAVE SAMPLE OUTPUT
# ===========================
sample_img = pairs[0][0]
depth = predict_depth(sample_img)

plt.imshow(depth, cmap="inferno")
plt.colorbar()
plt.title("Predicted Depth")
plt.savefig(os.path.join(OUTPUT_DIR, "sample_depth.png"))
plt.show()
