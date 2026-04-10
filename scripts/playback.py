import math
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from utils.hydra import next_run_index

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "device",
    lambda: "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu",
    replace=True,
)
OmegaConf.register_new_resolver(
    "next_index", lambda name, base: next_run_index(name, base), replace=True
)


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


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main playback loop for visualizing frames and feature maps."""
    run_dir = Path(cfg.run_dir)
    print(f"\nTarget run dir: {run_dir}\n")

    name = run_dir.name.lower()
    policy_name = cfg.get("policy", {}).get("name", "").lower()
    is_cnn = "cnn" in name or "cnn" in policy_name
    is_attn = "attn" in name or "attn" in policy_name

    viz_path = cfg.paths.viz if "paths" in cfg and "viz" in cfg.paths else "viz"
    frames_dir = run_dir / viz_path / "frames"
    fmap_dir = run_dir / viz_path / "feature_maps"

    if not frames_dir.exists():
        frames_dir = run_dir / "frames"
    if not fmap_dir.exists():
        fmap_dir = run_dir / "feature_maps"

    fps = 30

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

    out_path = run_dir / "playback.mp4"
    video_writer = None
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

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(out_path), fourcc, fps, (combined.shape[1], combined.shape[0])
            )

        video_writer.write(combined)
        any_shown = True

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {out_path}")

    if not any_shown:
        print("No frames were processed. Please check that feature maps exist and are processable.")


if __name__ == "__main__":
    main()
