# wabash-baseball-scouting-and-development

Tools for Wabash baseball scouting. Right now this is a baseball detector
that finds the ball in pitch video. Later it will feed ball tracking and
pitch-type classification.

## Setup (Windows, one time)

You need **Python 3.12** ([python.org](https://www.python.org/downloads/)),
**Git**, and ideally an **NVIDIA GPU**. Training on CPU works but is very slow.

```powershell
git clone https://github.com/ethankimmerle/wabash-baseball-scouting-and-development.git
cd wabash-baseball-scouting-and-development
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

The torch download is a few GB, so the install takes a while.

**Check that the GPU is visible:**

```powershell
venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

`True <your GPU name>` means you're set. `False` usually means either no
NVIDIA driver (update it from nvidia.com) or a CPU-only torch install
(re-run the pip install above, which uses the CUDA index).

## What's in here

| Path | What it is |
|---|---|
| [detection/](detection/) | Ball detector: training, re-splitting data by clip, and running a model on video. Start with [detection/README.md](detection/README.md). |
| [detection/LABELING.md](detection/LABELING.md) | Labeling rules everyone follows in Roboflow. **Read this before labeling.** |

## What isn't in git (on purpose)

Videos, Roboflow exports, training runs and model weights are too large for
git and are listed in `.gitignore`. Share them through OneDrive or Google
Drive:

- **Datasets:** re-export from Roboflow (project `baseball-pitch-detector`)
  and unzip into `detection/data/`.
- **Videos:** keep them anywhere and pass the path to
  `detection/predict_video.py`.
- **Trained models:** share the `best.pt` file directly. It's a single
  self-contained file.

## Working together

- Pull before you start (`git pull`) and commit small, focused changes.
- Don't commit videos, datasets or `.pt` files. `.gitignore` should catch
  them, but check `git status` before committing.
