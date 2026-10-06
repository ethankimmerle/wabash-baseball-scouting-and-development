# Detection

Trains a baseball detector locally (your own GPU, no ongoing API cost) from
a dataset labeled in Roboflow.

## The loop

1. **Label in Roboflow** (roboflow.com). Follow [LABELING.md](LABELING.md)
   exactly, and make sure everyone labeling uses the same rules. This is the
   step that actually determines how good the model gets; nothing
   downstream fixes bad or inconsistent labels.
2. **Export**: in your Roboflow project, Versions -> Generate New Version
   (filter out the `ambiguous` tag, no augmentations) -> Export -> format
   **YOLOv8** -> download the zip.
3. **Unzip it into `data/`** (this folder), so you end up with:
   ```
   detection/data/data.yaml
   detection/data/train/images/...
   detection/data/valid/images/...
   ```
4. **Re-split by clip**:
   ```
   venv\Scripts\python.exe detection\resplit_by_clip.py
   ```
   Roboflow splits randomly by frame, which puts near-identical frames from
   one clip in both train and valid and inflates validation scores. This
   regroups whole clips into train/valid/test and writes `data_by_clip/`.
   It needs at least 2 clips (3+ for a test set). Pin specific clips with
   `--valid-clips` / `--test-clips`.
5. **Train**:
   ```
   venv\Scripts\python.exe detection\train.py
   ```
   Uses `data_by_clip/` if it exists (otherwise `data/`, with a warning),
   fine-tunes `yolo26n` at 1280px on the GPU, and writes results to `runs/`.
   Look at validation **recall** first: it's how often the ball gets found.
6. **Use the result**: the trained model is
   `runs/baseball_detector*/weights/best.pt`. That single file is a
   complete, self-contained detector. Copy it anywhere, load it with
   `YOLO("best.pt")`, and run it on new frames with zero API calls.

## Why `data/`, `data_by_clip/` and `runs/` aren't in git

All three are in `.gitignore`. `data/` is a re-exportable copy of what's
really stored in Roboflow, and `data_by_clip/` is regenerated from it by the
re-split script. `runs/` is training output that piles up with every run;
regenerate it by re-running `train.py`. If you land on a model worth keeping
long-term, copy just that `best.pt` file somewhere intentional rather than
relying on `runs/`.
