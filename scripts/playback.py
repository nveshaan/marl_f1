import math
from pathlib import Path

import cv2
import numpy as np


def select_run(base_dir="experiments"):
    base_path = Path(base_dir)

    runs = sorted([p for p in base_path.iterdir() if p.is_dir()], key=lambda x: x.name)

    if len(runs) == 0:
        print("No runs found!")
        exit()

    print("\n=== Available runs ===\n")
    for i, run in enumerate(runs):
        print(f"[{i}] {run.name}")

    idx = int(input("\nSelect run index: "))
    return runs[idx]


def make_grid(fmap, variance_threshold=None):
    """Create a grid visualization from feature maps."""
    if fmap.ndim == 4:
        fmap = fmap[0]

    if fmap.ndim != 3:
        raise ValueError(f"Expected 3D feature map (C, H, W), got shape {fmap.shape}")

    C, H, W = fmap.shape
    grid_size = int(math.ceil(math.sqrt(C)))
    grid = np.zeros((grid_size * H, grid_size * W), dtype=np.uint8)

    for c in range(C):
        r = c // grid_size
        col = c % grid_size

        img = fmap[c]

        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-8:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        img = (img * 255).astype(np.uint8)

        grid[r * H : (r + 1) * H, col * W : (col + 1) * W] = img

    grid = cv2.applyColorMap(grid, cv2.COLORMAP_INFERNO)

    return grid


def main():
    """Main playback loop for visualizing frames and feature maps."""
    run_dir = select_run("experiments")
    print(f"\nSelected run: {run_dir}\n")

    name = run_dir.name.lower()
    is_cnn = "cnn" in name
    is_attn = "attn" in name

    frames_dir = run_dir / "viz" / "frames"
    fmap_dir = run_dir / "viz" / "feature_maps"

    if not frames_dir.exists():
        frames_dir = run_dir / "frames"
    if not fmap_dir.exists():
        fmap_dir = run_dir / "feature_maps"

    fps = 30
    delay = int(1000 / fps)

    screen_w, screen_h = 1280, 720  # fallback
    try:
        import tkinter as tk

        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception as e:
        print(f"Warning: Could not get screen size ({e}), using fallback")

    frame_files = sorted(frames_dir.glob("*.png"))

    if len(frame_files) == 0:
        print("No frames found!")
        exit()

    print(f"Found {len(frame_files)} frames")

    any_shown = False
    for _idx, frame_path in enumerate(frame_files):
        step = frame_path.stem.split("_")[-1]

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Frame {frame_path} could not be read, skipping.")
            continue

        fmap_img = None

        if is_cnn:
            fmap_files = sorted(fmap_dir.glob(f"*_{step}.npy"))

            if len(fmap_files) > 0:
                try:
                    fmap = np.load(str(fmap_files[0]))
                    fmap_img = make_grid(fmap)
                except Exception as e:
                    print(f"Step {step}: Could not process CNN feature map - {e}")
            else:
                print(f"Step {step}: No CNN feature map found, skipping frame.")

        elif is_attn:
            fmap_files = sorted(fmap_dir.glob(f"*attn_{step}.npy"))

            if len(fmap_files) > 0:
                attn = np.load(str(fmap_files[0]))
                if attn.ndim == 4:
                    attn = attn[0]

                if attn.ndim == 3:
                    fmap_img = make_grid(attn)
                elif attn.ndim == 2:
                    attn_img = attn
                    attn_img = (attn_img - attn_img.min()) / (attn_img.max() + 1e-8)
                    attn_img = (attn_img * 255).astype(np.uint8)
                    fmap_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_INFERNO)
                else:
                    print(
                        f"Step {step}: Attention map has unexpected ndim {attn.ndim} after slicing, skipping frame."
                    )
                    continue
            else:
                print(f"Step {step}: No attention feature map found, skipping frame.")

        if fmap_img is None:
            print(f"Step {step}: No feature map image generated, skipping frame.")
            continue

        scale = frame.shape[0] / fmap_img.shape[0]
        fmap_img = cv2.resize(fmap_img, (int(fmap_img.shape[1] * scale), frame.shape[0]))

        combined = np.hstack([frame, fmap_img])

        h, w = combined.shape[:2]
        scale = min(screen_w / w, screen_h / h, 1.0)
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

        cv2.imshow("Playback: Frame | Feature Grid", combined)
        any_shown = True

        key = cv2.waitKey(delay)
        if key == ord("q"):
            break

    if not any_shown:
        print("No frames were displayed. Please check that feature maps exist and are processable.")
        # Keep window open for user to see message
        import time

        time.sleep(3)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
