# Labeling rules

Everyone labeling this dataset follows these rules. Labeling the same way
matters more than exactly where the cutoff is.

**Class:** `baseball` (the only class).

## Every frame goes in one of three buckets

| What you see | What to do |
|---|---|
| **Ball is visible**, even as a blurry streak or partly hidden | **Label it.** Draw the box around the *whole* blur streak, not just where you think the center is. |
| **Definitely no ball** (before release, after the catch, empty mound) | **Keep it with no boxes.** These teach the model to ignore heads, white uniform patches and gloves. Aim for about 10–15% of the dataset. |
| **Can't tell**, or you only "know" it's there from the frames before and after | **Tag it `ambiguous` and draw nothing.** It gets filtered out at export, so it's never used for training. |

**Only mark a frame "no ball" when you're sure.** If a faint ball is in a
frame left blank, the model learns that a faint ball is background, which
hurts it on exactly the hard frames we care about. When in doubt, tag it
`ambiguous`.

It's fine for the detector to miss a frame here and there. The SORT tracker
downstream bridges a few missed frames. False detections are what hurt,
because they spawn extra tracks. So a careful detector beats an eager one.

## Boxes
- The box is tight around everything that is the ball, blur included.
- One box per ball. Don't split a streak into pieces.
- Don't box gloves, mitts, rosin bags or other look-alikes. Frames with them
  and no ball are useful background frames.

## Clips and naming
- Keep the frame names as extracted (`<clip>_frame_<n>.jpg`, or the `<clip>_mp4-<n>` names Roboflow gives uploaded videos). The re-split
  script uses the part before the frame number to keep whole clips together.
- Label frames from **many different clips** (pitchers, lighting, camera
  angles, day/night). Variety helps more than extra frames from one clip.

## Exporting (from Roboflow)
Go to Versions → Generate New Version:
- **Filter out the `ambiguous` tag.**
- **Augmentations: none.** Ultralytics does its own during training.
- Any train/valid/test split is fine, because `resplit_by_clip.py` redoes it.

Then Export → **YOLOv8** → download the zip → unzip into `detection/data/`.

> Free-plan Roboflow projects are public. Confirm the team is OK with the
> footage being visible before uploading more.
