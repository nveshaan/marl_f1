import cv2
import numpy as np
from pathlib import Path
import math

# ==============================
# SELECT RUN
# ==============================
def select_run(base_dir="experiments"):
    base_path = Path(base_dir)

    runs = [p for p in base_path.iterdir() if p.is_dir()]

    if len(runs) == 0:
        print("No runs found!")
        exit()

    print("\nAvailable runs:\n")
    for i, run in enumerate(runs):
        print(f"[{i}] {run.name}")

    idx = int(input("\nSelect run index: "))
    return runs[idx]


run_dir = select_run("experiments")
print(f"\nSelected run: {run_dir}\n")

# ==============================
# DETECT POLICY TYPE
# ==============================
name = run_dir.name.lower()
is_cnn = "cnn" in name
is_attn = "attn" in name

# ==============================
# PATHS
# ==============================
frames_dir = run_dir / "feature_viz" / "frames"
fmap_dir = run_dir / "feature_viz" / "feature_maps"

if not frames_dir.exists():
    frames_dir = run_dir / "frames"
if not fmap_dir.exists():
    fmap_dir = run_dir / "feature_maps"

fps = 30
delay = int(1000 / fps)

# ==============================
# SCREEN SIZE
# ==============================
screen_w, screen_h = 1280, 720  # fallback
try:
    import tkinter as tk
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
except:
    pass

# ==============================
# LOAD FILE LIST
# ==============================
frame_files = sorted(frames_dir.glob("*.png"))

if len(frame_files) == 0:
    print("No frames found!")
    exit()

print(f"Found {len(frame_files)} frames")

# ==============================
# GRID FUNCTION
# ==============================
def make_grid(fmap):
    import math
    C, H, W = fmap.shape

    grid_size = int(math.ceil(math.sqrt(C)))
    grid = np.zeros((grid_size * H, grid_size * W), dtype=np.uint8)

    for i in range(C):
        r = i // grid_size
        c = i % grid_size

        img = fmap[i]

        # normalize per channel
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-8:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        img = (img * 255).astype(np.uint8)

        grid[r*H:(r+1)*H, c*W:(c+1)*W] = img

    # 🔥 apply plasma colormap here
    grid = cv2.applyColorMap(grid, cv2.COLORMAP_PLASMA)

    return grid

# ==============================
# PLAYBACK LOOP
# ==============================
for frame_path in frame_files:
    step = frame_path.stem.split("_")[-1]

    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue

    fmap_img = None

    if is_cnn:
        fmap_files = sorted(fmap_dir.glob(f"*_{step}.npy"))

        if len(fmap_files) > 0:
            fmap = np.load(str(fmap_files[0]))
            fmap_img = make_grid(fmap)

    elif is_attn:
        fmap_files = sorted(fmap_dir.glob(f"*_{step}.npy"))

        if len(fmap_files) > 0:
            attn = np.load(str(fmap_files[0]))

            if attn.ndim == 3:
                attn_img = attn.mean(0)
            elif attn.ndim == 2:
                attn_img = attn
            else:
                continue

            attn_img = (attn_img - attn_img.min()) / (attn_img.max() + 1e-8)
            attn_img = (attn_img * 255).astype(np.uint8)
            fmap_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_JET)

    if fmap_img is None:
        continue

    # resize fmap to match frame height
    scale = frame.shape[0] / fmap_img.shape[0]
    fmap_img = cv2.resize(fmap_img, (int(fmap_img.shape[1]*scale), frame.shape[0]))

    combined = np.hstack([frame, fmap_img])

    # resize to screen
    h, w = combined.shape[:2]
    scale = min(screen_w / w, screen_h / h, 1.0)
    combined = cv2.resize(combined, (int(w*scale), int(h*scale)))

    cv2.imshow("Playback: Frame | Feature Grid", combined)

    key = cv2.waitKey(delay)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
