import fnmatch
import importlib
import shutil
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from agents import BaseAgent

importlib.import_module("multi_car_racing")
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))
OmegaConf.register_new_resolver(
    "device",
    lambda: "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu",
    replace=True,
)


def select_run(base_dir="experiments"):
    """Prompt user to select a completed run from experiments directory."""
    base_path = Path(base_dir)

    runs = sorted(
        [p for p in base_path.iterdir() if p.is_dir() and (p / "best_model").exists()],
        key=lambda x: x.name,
    )

    if not runs:
        raise ValueError(f"No completed runs found in {base_dir}")

    print("\n=== Available runs ===\n")
    for i, run in enumerate(runs):
        print(f"[{i}] {run.name}")

    idx = int(input("\nSelect run index: "))
    return runs[idx]


def main():
    run_dir = select_run("experiments")
    print(f"\nSelected run: {run_dir}\n")

    viz_dir = run_dir / "viz"
    if viz_dir.exists():
        shutil.rmtree(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg.run_dir = str(run_dir)
    OmegaConf.resolve(cfg)

    policy_name = cfg.get("policy", {}).get("name", "unknown")
    print(f"\n=== Model Policy: {policy_name} ===\n")

    print("Instantiating agent...")
    agent: BaseAgent = instantiate(cfg.algo.agent, cfg=cfg, _recursive_=False)

    print("\nDiscovering available layers...")
    available_layers = agent.get_available_layers()

    if not available_layers:
        print("No hookable layers found. Proceeding with empty layer selection.")
        selected_layers = []
    else:
        print("\n=== Available Layers ===")
        for i, (name, _) in enumerate(available_layers):
            print(f"[{i}] {name}")
        print("\nSelect layers to save (use patterns or indices):")
        print('  Examples: "all", "relu_*", "0,2,4", "attn_0,attn_2"')
        selection = input("\nYour selection: ").strip()
        if selection.lower() == "all":
            selected_layers = [name for name, _ in available_layers]
        else:
            selected = []
            layer_names = [name for name, _ in available_layers]
            for part in selection.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 0 <= idx < len(layer_names):
                        selected.append(layer_names[idx])
                else:
                    matched = fnmatch.filter(layer_names, part)
                    selected.extend(matched)
            selected_layers = list(set(selected))

    num_steps = None
    steps_input = input("\nMax steps (press Enter for full episode): ").strip()
    if steps_input.isdigit():
        num_steps = int(steps_input)

    print("\nStarting evaluation with feature visualization...")
    agent.eval(num_steps=num_steps, deterministic=True, selected_layers=selected_layers)


if __name__ == "__main__":
    main()
