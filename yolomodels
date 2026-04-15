# ==========================================
# UNIFIED VISION PIPELINE
# YOLOv8 + YOLOv11 + RT-DETR + MiDaS Depth
# ==========================================

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ultralytics (YOLOv8 / YOLOv11 / RT-DETR)
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
IMAGE_PATH = "sample.jpg"   # input image
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# LOAD MODELS
# ==========================================
print("Loading models...")

# YOLOv12
yolo12 = YOLO("yolov12n.pt")

# YOLOv8
yolo8 = YOLO("yolov8n.pt")

# YOLOv11 (latest ultralytics naming)
yolo11 = YOLO("yolo11n.pt")

# RT-DETR
rtdetr = YOLO("rtdetr-l.pt")

# MiDaS Depth
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(DEVICE)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
depth_transform = transforms.small_transform

print("All models loaded!")

# ==========================================
# DEPTH ESTIMATION
# ==========================================
def predict_depth(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = depth_transform(img).to(DEVICE)

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


# ==========================================
# OBJECT DETECTION
# ==========================================
def run_detection(model, image, name):
    results = model(image)[0]

    annotated = results.plot()

    save_path = os.path.join(OUTPUT_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, annotated)

    return results, annotated


# ==========================================
# VISUALIZATION
# ==========================================
def save_depth(depth):
    plt.imshow(depth, cmap="inferno")
    plt.colorbar()
    plt.title("Depth Map")
    plt.savefig(os.path.join(OUTPUT_DIR, "depth.png"))
    plt.close()


# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Error: Image not found!")
        return

    print("Running YOLOv8...")
    res8, img8 = run_detection(yolo8, image, "yolo8")

    print("Running YOLOv11...")
    res11, img11 = run_detection(yolo11, image, "yolo11")

    print("Running RT-DETR...")
    res_rt, img_rt = run_detection(rtdetr, image, "rtdetr")

    print("Running Depth Estimation...")
    depth = predict_depth(image)
    save_depth(depth)

    print("Pipeline completed!")
    print(f"Results saved in: {OUTPUT_DIR}")


# ==========================================
# BENCHMARK (OPTIONAL)
# ==========================================
def benchmark(image_folder):
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]

    print("\nRunning Benchmark...\n")

    for img_path in tqdm(images):
        img = cv2.imread(img_path)

        _ = yolo8(img)
        _ = yolo11(img)
        _ = rtdetr(img)
        _ = predict_depth(img)

    print("Benchmark completed!")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
