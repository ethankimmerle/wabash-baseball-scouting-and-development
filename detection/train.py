"""
Train a baseball detector from a Roboflow-exported (YOLOv8 format) dataset.

THE WHOLE PIPELINE THIS FITS INTO:
    1. Label images in Roboflow (roboflow.com) -- draw boxes around the ball,
       and just as importantly, label things that AREN'T a ball but look
       like one (a glove, a mitt) as hard negatives.
    2. In Roboflow: Versions -> Generate New Version -> Export -> "YOLOv8"
       format -> download the zip.
    3. Unzip it into detection/data/ (next to this file), so you end up
       with detection/data/data.yaml, detection/data/train/, etc.
    3b. Run resplit_by_clip.py so whole clips (not random frames) land in
       train/valid/test. This script uses data_by_clip/ when it exists.
    4. Run this script. It trains a small YOLO model on your GPU (or CPU,
       just slower) and saves the result under detection/runs/.
    5. The trained weights (a .pt file) are YOURS -- no API, no per-call
       cost, run them locally forever.

WHY YOLO, AND WHY THE "nano" SIZE:
    YOLO (You Only Look Once) is a well-established, well-documented object
    detection architecture -- not exotic, tons of tutorials/community
    support if you get stuck. "yolo26n.pt" below is the smallest ("nano")
    pretrained checkpoint: fast to train, fast to run, and plenty capable
    for a single-class problem like "find the baseball". You're not
    training from zero -- yolo26n.pt already knows general "what does an
    object look like" from millions of images (COCO dataset); this script
    fine-tunes that existing knowledge onto your baseball images
    specifically. That's why a small dataset can still produce a usable
    model, and why training takes minutes, not days.

WHAT "EPOCHS" MEANS:
    One epoch = the model has seen every training image once. More epochs
    generally = better fit, up to a point where it starts memorizing your
    specific training images instead of learning the general pattern
    ("overfitting"). We allow up to 150 but set PATIENCE: if validation
    hasn't improved for that many epochs, training stops early and keeps
    the best checkpoint, so a high cap costs nothing.

WHY 1280 AND NOT THE DEFAULT 640:
    The ball is only a handful of pixels in a full frame. At 640 it gets
    downscaled to almost nothing. 1280 keeps it visible, at the cost of
    ~4x memory, hence the smaller batch. If you get a CUDA out-of-memory
    error, drop BATCH to 4. If the ball is still missed at 1280, the next
    step is tiled inference (SAHI), not a bigger model.
"""

import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_SPLIT_YAML = os.path.join(HERE, "data_by_clip", "data.yaml")  # from resplit_by_clip.py
RAW_EXPORT_YAML = os.path.join(HERE, "data", "data.yaml")          # straight from Roboflow

# yolo26n: newest nano checkpoint in Ultralytics, with improvements aimed at
# small objects. Uses the copy in weights/ if present, otherwise downloads it.
LOCAL_WEIGHTS = os.path.join(HERE, "..", "weights", "yolo26n.pt")
BASE_MODEL = LOCAL_WEIGHTS if os.path.exists(LOCAL_WEIGHTS) else "yolo26n.pt"

EPOCHS = 150
PATIENCE = 30    # stop early if validation hasn't improved in this many epochs
IMAGE_SIZE = 1280  # ball is tiny; 640 shrinks it to a few pixels
BATCH = 8        # fits an 8 GB RTX 4060 at 1280; drop to 4 on out-of-memory


def main():
    if os.path.exists(CLIP_SPLIT_YAML):
        data_yaml = CLIP_SPLIT_YAML
    elif os.path.exists(RAW_EXPORT_YAML):
        data_yaml = RAW_EXPORT_YAML
        print("WARNING: training on Roboflow's random frame split. Validation scores will look")
        print("better than reality. Run resplit_by_clip.py once you have 2+ clips.\n")
    else:
        raise FileNotFoundError(
            f"Couldn't find {RAW_EXPORT_YAML}.\n"
            "Export your dataset from Roboflow (YOLOv8 format) and unzip it "
            "into detection/data/ -- see the top of this file for the exact steps."
        )
    print(f"Dataset: {data_yaml}")

    # Start from a small COCO-pretrained checkpoint rather than random
    # weights -- this is "transfer learning": the model already knows
    # general object-detection basics, we're just specializing it.
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=0,        # the GPU; training on CPU at 1280 would take forever
        mosaic=1.0,      # stitch 4 images together -- more ball positions/scales per batch
        fliplr=0.5,      # mirror left/right (lefty vs righty pitchers)
        project=os.path.join(HERE, "runs"),
        name="baseball_detector",
    )

    print("\nTraining complete.")
    print(f"Best weights saved to: {results.save_dir / 'weights' / 'best.pt'}")
    print("Check validation RECALL first -- that's the number that says how often the ball is found.")
    print("That .pt file is the model -- copy it wherever you want to run inference from.")


if __name__ == "__main__":
    main()
