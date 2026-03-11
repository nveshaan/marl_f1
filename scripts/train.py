import pathlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from agents import BaseAgent

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.joinpath("configs")))
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    run_dir = Path()
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    mntr_dir = run_dir / "monitor"
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mntr_dir.mkdir(parents=True, exist_ok=True)

    cls = hydra.utils.get_class(cfg._target_)
    agent: BaseAgent = cls(cfg)
    agent.learn()


if __name__ == "__main__":
    main()
