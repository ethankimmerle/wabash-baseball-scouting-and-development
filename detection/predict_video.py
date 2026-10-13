"""
Run a trained baseball detector over a video and save a copy with the
detections drawn on it -- the quickest way to SEE how the model is doing.

Validation numbers tell you how often the model is right; this tells you
HOW it's wrong (missing the ball mid-flight? firing on the catcher's mitt?
only catching it near the plate?), which is what decides what to label next.

USAGE:
    venv\\Scripts\\python.exe detection\\predict_video.py path\\to\\clip.mp4
    venv\\Scripts\\python.exe detection\\predict_video.py clip.mp4 --conf 0.1
    venv\\Scripts\\python.exe detection\\predict_video.py clip.mp4 --weights detection\\runs\\baseball_detector-2\\weights\\best.pt

By default it uses the best.pt from the most recent training run. Each model
gets its own output folder, so results from different models never
overwrite each other and can be compared side by side:

    detection/runs/predict/<model run>/<clip>_conf<conf>.mp4
    detection/runs/predict/<model run>/summary.csv   (one row per video run)

e.g. runs/predict/baseball_detector-4/Gaskill_TEST_conf0.10.mp4

WHAT --conf DOES:
    Every detection comes with a confidence score (0-1). Only boxes at or
    above --conf are drawn. Lower it (e.g. 0.1) to see what the model is
    "almost" detecting -- useful for an early model that's unsure of itself.
    Raise it to see only what the model is sure about.
"""

import argparse
import csv
import glob
import os

import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "runs", "predict")


def latest_weights():
    candidates = glob.glob(os.path.join(HERE, "runs", "*", "weights", "best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained model found under detection/runs/ -- run train.py first.")
    return max(candidates, key=os.path.getmtime)


def model_name(weights):
    """'runs/baseball_detector-4/weights/best.pt' -> 'baseball_detector-4'."""
    weights = os.path.abspath(weights)
    parent = os.path.dirname(weights)
    if os.path.basename(parent) == "weights":
        return os.path.basename(os.path.dirname(parent))
    return os.path.splitext(os.path.basename(weights))[0]  # a .pt copied somewhere else


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="path to the video to run the detector on")
    parser.add_argument("--weights", default=None, help="model .pt file (default: newest runs/*/weights/best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence to draw a box (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference size; match what you trained at")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (quick tests)")
    args = parser.parse_args()

    weights = args.weights or latest_weights()
    print(f"Model: {weights}")
    model = YOLO(weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Couldn't open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.join(OUT_DIR, model_name(weights))
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(out_dir, f"{name}_conf{args.conf:.2f}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = 0
    frames_with_ball = 0
    confidences = []
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frames >= args.max_frames):
            break
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            confidences.append(conf)
        if len(result.boxes):
            frames_with_ball += 1

        # Frame number in the corner so you can note exactly which frames fail.
        cv2.putText(frame, f"frame {frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        frames += 1
        if frames % 100 == 0:
            print(f"  {frames}/{total} frames")

    cap.release()
    writer.release()

    print(f"\nProcessed {frames} frames; {frames_with_ball} had at least one detection (conf >= {args.conf}).")
    if confidences:
        print(f"Confidence of drawn boxes: min {min(confidences):.2f}, max {max(confidences):.2f}, "
              f"mean {sum(confidences) / len(confidences):.2f}")
    print(f"Saved: {out_path}")

    # One row per run, so models can be compared at a glance without
    # rewatching every video.
    summary = os.path.join(out_dir, "summary.csv")
    new_file = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["clip", "conf", "frames", "frames_with_detection", "max_conf", "mean_conf"])
        w.writerow([name, args.conf, frames, frames_with_ball,
                    f"{max(confidences):.2f}" if confidences else "",
                    f"{sum(confidences) / len(confidences):.2f}" if confidences else ""])


if __name__ == "__main__":
    main()
