"""
Re-split a Roboflow export so whole CLIPS land in train/valid/test, never
individual frames.

WHY THIS EXISTS:
    Roboflow's random split scatters frames from the same clip across
    train and valid. Neighboring frames look almost identical, so the model
    gets "validated" on images it has basically already trained on, and the
    scores come out falsely high. Splitting by clip means validation is
    measured on footage the model has genuinely never seen.

HOW CLIPS ARE IDENTIFIED:
    From the filename -- everything before the frame number is the clip ID.
    Two naming styles are recognized:
      "<clip>_frame_001_jpg.rf.<hash>.jpg"  (frames extracted yourself)
      "<clip>_mp4-0001_jpg.rf.<hash>.jpg"   (video uploaded straight to Roboflow)

WHAT IT DOES:
    Pools every image+label from data/ (all three splits), groups them by
    clip, assigns whole clips to train/valid/test aiming for the target
    ratios, and COPIES the result into data_by_clip/ with its own data.yaml.
    data/ is left untouched, so this is safe to re-run after every export.

USAGE:
    venv\\Scripts\\python.exe detection\\resplit_by_clip.py
    venv\\Scripts\\python.exe detection\\resplit_by_clip.py --valid-clips CLIP_A --test-clips CLIP_B
      (pin specific clips to valid/test; everything else goes to train)
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data")
DST = os.path.join(HERE, "data_by_clip")
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLIP_RE = re.compile(r"^(.*?)(?:_frame_|_(?:mp4|mov|avi|mkv|m4v)-)\d+", re.IGNORECASE)


def clip_id(filename):
    match = CLIP_RE.match(filename)
    if not match:
        raise ValueError(
            f"Can't find a clip ID in '{filename}' (expected '<clip>_frame_<n>...' or '<clip>_mp4-<n>...'). "
            "Add the pattern to CLIP_RE so clips can be grouped."
        )
    return match.group(1)


def collect_frames():
    """Return {clip_id: [(image_path, label_path_or_None), ...]} across all splits."""
    clips = defaultdict(list)
    for split in SPLITS:
        img_dir = os.path.join(SRC, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            label = os.path.join(SRC, split, "labels", os.path.splitext(name)[0] + ".txt")
            clips[clip_id(name)].append(
                (os.path.join(img_dir, name), label if os.path.exists(label) else None)
            )
    return clips


def is_labeled(label_path):
    return label_path is not None and os.path.getsize(label_path) > 0


def assign_clips(clips, valid_ratio, test_ratio, seed, pinned):
    """Greedy: biggest clips first, each goes to the split furthest below its target."""
    assignment = dict(pinned)
    free = [c for c in clips if c not in assignment]
    total = sum(len(frames) for frames in clips.values())
    targets = {"train": 1 - valid_ratio - test_ratio, "valid": valid_ratio, "test": test_ratio}
    # With few clips there isn't enough to fill all three; drop test first.
    if len(clips) < 3:
        targets["test"] = 0

    counts = {s: 0 for s in SPLITS}
    for c, s in assignment.items():
        counts[s] += len(clips[c])

    rng = random.Random(seed)
    rng.shuffle(free)
    free.sort(key=lambda c: len(clips[c]))  # smallest first; ties keep the shuffled order

    def place(c, split):
        assignment[c] = split
        counts[split] += len(clips[c])

    # Make sure valid and test each get at least one clip, using the smallest
    # ones so the big clips stay available for training.
    for split in ("valid", "test"):
        if targets[split] > 0 and counts[split] == 0 and len(free) > 1:
            place(free.pop(0), split)

    # Then biggest-first, each clip to whichever split is furthest below target.
    for c in reversed(free):
        place(c, max(SPLITS, key=lambda s: targets[s] * total - counts[s]))
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="change to get a different clip assignment")
    parser.add_argument("--valid-clips", nargs="*", default=[], help="clip IDs to force into valid")
    parser.add_argument("--test-clips", nargs="*", default=[], help="clip IDs to force into test")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(SRC, "data.yaml")):
        raise FileNotFoundError(f"No Roboflow export found at {SRC} -- unzip the YOLOv8 export there first.")

    clips = collect_frames()
    print(f"Found {sum(len(f) for f in clips.values())} frames across {len(clips)} clip(s).")
    if len(clips) < 2:
        raise SystemExit(
            "Need at least 2 clips to split by clip (3+ to also get a test set).\n"
            "Label and export frames from more clips, then re-run."
        )

    pinned = {c: "valid" for c in args.valid_clips}
    pinned.update({c: "test" for c in args.test_clips})
    unknown = [c for c in pinned if c not in clips]
    if unknown:
        raise SystemExit(f"Unknown clip ID(s): {unknown}. Known clips: {sorted(clips)}")

    assignment = assign_clips(clips, args.valid_ratio, args.test_ratio, args.seed, pinned)

    if os.path.exists(DST):
        shutil.rmtree(DST)
    stats = {s: {"clips": 0, "frames": 0, "labeled": 0} for s in SPLITS}
    for c, split in assignment.items():
        os.makedirs(os.path.join(DST, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DST, split, "labels"), exist_ok=True)
        stats[split]["clips"] += 1
        for img, label in clips[c]:
            shutil.copy2(img, os.path.join(DST, split, "images"))
            label_dst = os.path.join(DST, split, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
            if label:
                shutil.copy2(label, label_dst)
            else:
                open(label_dst, "w").close()  # no label file = background frame
            stats[split]["frames"] += 1
            stats[split]["labeled"] += is_labeled(label)

    # Reuse class names from the original export so they can't drift.
    with open(os.path.join(SRC, "data.yaml")) as f:
        names_lines = [line for line in f if line.startswith(("nc:", "names:"))]
    splits_present = [s for s in SPLITS if stats[s]["frames"]]
    with open(os.path.join(DST, "data.yaml"), "w") as f:
        f.write(f"path: {DST}\n")
        for s in splits_present:
            f.write(f"{'val' if s == 'valid' else s}: {s}/images\n")
        f.write("\n" + "".join(names_lines))

    print(f"\nWrote {DST}")
    for s in SPLITS:
        st = stats[s]
        if not st["frames"]:
            continue
        bg = st["frames"] - st["labeled"]
        print(f"  {s:5}  {st['clips']:3} clips  {st['frames']:5} frames  "
              f"{st['labeled']:5} with ball  {bg:5} background ({bg / st['frames']:.0%})")
    print("\nClip assignment:")
    for c in sorted(assignment, key=lambda c: (SPLITS.index(assignment[c]), c)):
        print(f"  {assignment[c]:5}  {c}  ({len(clips[c])} frames)")


if __name__ == "__main__":
    main()
