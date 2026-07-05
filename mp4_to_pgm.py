import cv2
import glob
import os
from ultralytics import YOLO

VIDEO_PATH     = "digit_video.mp4"
YOLO_MODEL     = "runs/classify/yolo_train/digit_cls-3/weights/best.pt"
OUT_DIR        = "mnistCUDNN/pgm_output"
CONF_THRESH    = 0.7
CONFIRM_FRAMES = 15

os.makedirs(OUT_DIR, exist_ok=True)
for f in glob.glob(os.path.join(OUT_DIR, "*.pgm")):
    os.remove(f)

model = YOLO(YOLO_MODEL)

def frame_to_pgm(frame, save_path):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray    = cv2.bitwise_not(gray)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, resized)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    current_digit = None
    consecutive   = 0
    confirmed     = False
    best_frame    = None
    best_conf     = 0.0
    pgm_idx       = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)
        top1   = result[0].probs.top1
        conf   = float(result[0].probs.top1conf)
        label  = result[0].names[top1] if conf >= CONF_THRESH else None

        if label is None:
            current_digit = None
            consecutive   = 0
            confirmed     = False
            best_frame    = None
            best_conf     = 0.0
        elif label == current_digit:
            consecutive += 1
            if conf > best_conf:
                best_conf  = conf
                best_frame = frame.copy()
        else:
            current_digit = label
            consecutive   = 1
            best_conf     = conf
            best_frame    = frame.copy()
            confirmed     = False

        if consecutive == CONFIRM_FRAMES and not confirmed:
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            pgm_name  = f"frame_{frame_num:06d}_digit_{label}_conf_{best_conf:.2f}_{pgm_idx:04d}.pgm"
            frame_to_pgm(best_frame, os.path.join(OUT_DIR, pgm_name))
            print(f"Saved: {pgm_name}")
            pgm_idx  += 1
            confirmed = True

    cap.release()
    print(f"\nTotal PGM files saved: {pgm_idx - 1}")

process_video(VIDEO_PATH)
