import importlib
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from agents import BaseAgent
from utils.run_index import next_run_index

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


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    for path in cfg.paths.values():
        (run_dir / path).mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.run_dir = str(run_dir)

    agent: BaseAgent = instantiate(cfg.algo.agent, cfg=cfg, _recursive_=False)
    agent.learn()
    agent.save(run_dir)


if __name__ == "__main__":
    main()
