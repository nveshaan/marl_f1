# F1 Multi-Agent Reinforcement Learning

**Project Page:** https://nveshaan.github.io/projects/marl-f1/

<!-- BEGIN:PROJECT_TREE -->
```text
marl_f1/
├── agents/
│   └── base_agent.py
├── configs/
│   └── train.yaml
├── models/
├── multi_car_racing/
├── scripts/
│   ├── eval.py
│   ├── playback.py
│   ├── plots.py
│   └── train.py
├── utils/
│   └── hydra.py
├── .gitignore
├── .gitmodules
├── LICENSE
├── main.py
├── pyproject.toml
└── README.md
```
<!-- END:PROJECT_TREE -->

## Setup

```bash
git clone --recurse-submodules https://github.com/nveshaan/marl_f1.git
cd marl_f1
uv sync
```

To update the cloned repo,

```bash
git pull
git submodule update

# if you are contributing to submodule as well,
cd multi_car_racing
git checkout master
git pull

uv sync
```

## Training

```bash
python -m scripts.train
```

| Argument | Description                                                           | Default  | Available values                                      |
| -------- | --------------------------------------------------------------------- | -------- | ----------------------------------------------------- |
| `task`   | Experiment tag/profile used in naming.                                |  |                  |
| `algo`   | RL algorithm config, including model settings and training timesteps. |    |  |
| `policy` | Policy architecture/config used by the selected algorithm.            |     |                                          |
| `agent`  | Environment wrapper/vectorization backend.                            |     |                                            |
| `seed`   | Global random seed used for train and eval environments.              | `42`     | Any integer                                           |

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
