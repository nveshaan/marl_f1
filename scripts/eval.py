import datetime
import importlib
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agents import BaseAgent
from utils.hydra import next_run_index

importlib.import_module("multi_car_racing")

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


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    cfg.env_kwargs = {**cfg.task.env_kwargs, **cfg.algo.env_kwargs}
    cfg.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = Path(cfg.run_dir)
    for path in cfg.paths.values():
        (run_dir / path).mkdir(parents=True, exist_ok=True)

    best_model_dir = run_dir / cfg.paths.best_model
    best_model_zip = best_model_dir / "best_model.zip"
    worldmodel_pt = best_model_dir / "worldmodel.pt"

    if best_model_zip.exists():
        load_path = str(best_model_zip)
    elif worldmodel_pt.exists():
        load_path = str(best_model_dir)
    else:
        print(f"Error: Best model not found in {best_model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Instantiating agent from {cfg.algo.name}...")
    agent: BaseAgent = instantiate(cfg.algo.agent, cfg=cfg, _recursive_=False)

    print(f"Loading best model from {load_path}...")
    agent.load(load_path)

    num_steps = cfg.get("num_steps", None)
    selected_layers = cfg.get("selected_layers", None)

    print("\nStarting evaluation with loaded best model...")
    agent.eval(num_steps=num_steps, deterministic=True, selected_layers=selected_layers)


if __name__ == "__main__":
    main()
