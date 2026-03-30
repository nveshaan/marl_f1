# F1 Multi-Agent Reinforcement Learning

**Project Page:** https://nveshaan.github.io/projects/marl-f1/

<!-- BEGIN:PROJECT_TREE -->
```text
marl_f1/
├── agents/
│   ├── common/
│   │   ├── bmarl.py
│   │   └── sb3.py
│   ├── base_agent.py
│   └── baseline.py
├── configs/
│   └── train.yaml
├── models/
│   └── selection_attention_extractor.py
├── multi_car_racing/
├── scripts/
│   ├── eval.py
│   ├── playback.py
│   └── train.py
├── utils/
│   └── run_index.py
├── .gitignore
├── .gitmodules
├── AUTHORS
├── LICENSE
├── main.py
├── pyproject.toml
└── README.md
```
<!-- END:PROJECT_TREE -->

## Setup

> This project uses `uv` as the package/environment manager. Install it from [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone --recurse-submodules https://github.com/nveshaan/marl_f1.git
cd marl_f1
uv sync
```

To update the cloned repo,
```bash
git pull
git submodule update

# if you are contributing to submodules as well,
cd {submodule}
git checkout main
git pull

uv sync
```

To play with notebooks,
```bash
uv sync --group notebook
```

## Training

```bash
python -m scripts.train --multirun algo=dqn seed=42,43
```

| Argument    | Description                                                             | Default       | Available values        |
| ----------- | ----------------------------------------------------------------------- | ------------- | ----------------------- |
| `task`      | Experiment tag/profile used in naming.                                  | `single`      | ex. `competitive`       |
| `algo`      | RL algorithm config, including model settings and training timesteps.   | `dqn`         | `dqn`, `sac`            |
| `policy`    | Policy architecture/config used by the selected algorithm.              | `cnn`         | `cnn`                   |
| `seed`      | Global random seed used for train and eval environments.                | `42`          | Any integer             |

Hydra-style overrides are supported, so you can also set additional fields from `configs/train.yaml`.

To see training logs,
```bash
tensorboard --logdir ./experiments
```

To evaluate and visualize feature maps,
```bash
python -m scripts.eval
python -m scripts.playback
```

## Contributing

To contribute to this repository, setup the `.venv` as follows,

```bash
uv sync --group dev
uv run --group dev pre-commit install
uv run --group dev pre-commit run --all-files
```

This might make your commits fail due to ruff checks, please consider the changes and re-commit.

## Acknowledgements

This project builds on ideas and implementations from prior work in multi-agent reinforcement learning, including:

- Schwarting, W., Seyde, T., Gilitschenski, I., Liebenwein, L., Sander, R., Karaman, S., and Rus, D. (2020). _Deep Latent Competition: Learning to Race Using Visual Control Policies in Latent Space_. Conference on Robot Learning (CoRL 2020). https://arxiv.org/abs/2102.09812
- Ha, D., and Schmidhuber, J. (2018). _World Models_. arXiv:1803.10122. https://arxiv.org/abs/1803.10122

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
