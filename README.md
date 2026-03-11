# F1 Multi-Agent Reinforcement Learning
**Project Page:** https://nveshaan.github.io/projects/marl-f1/
<!-- BEGIN:PROJECT_TREE -->
```text
marl_f1/
├── agents/
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── competitive_agent.py
│   │   ├── cooperative_agent.py
│   │   ├── mixed_agent.py
│   │   └── single_agent.py
│   ├── world_models/
│   │   └── __init__.py
│   ├── __init__.py
│   └── base_agent.py
├── configs/
│   └── single_agent.yml
├── models/
├── multi_car_racing/
│   ├── gym_multi_car_racing/
│   │   ├── __init__.py
│   │   └── multi_car_racing.py
│   ├── .git
│   ├── .gitignore
│   ├── AUTHORS
│   ├── LICENSE
│   ├── pyproject.toml
│   └── README.md
├── scripts/
│   ├── eval.py
│   └── train.py
├── utils/
│   └── update_readme_tree.py
├── .gitignore
├── .gitmodules
├── .pre-commit-config.yaml
├── .python-version
├── LICENSE
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

# update the cloned repo
git pull
git submodule update
uv sync

# play with notebooks
uv sync --group notebook
```

## Training
```bash
python scripts/train.py CarRacing-v0
```

## Contributing
To contribute to this repository, setup the `.venv` as follows,
```bash
uv sync --group dev
uv run --group dev pre-commit install
uv run --group dev pre-commit run --all-files
```
Do note that if there is change in the directory structure (addition or removal of files/folders), `utils/update_readme_tree.py` will make `git commit` throw an error. Kindly ignore and `git commit -a README.md`. It is just a `pre-commit` script to update the directory tree in `README.md`

## Acknowledgements

This project builds on ideas and implementations from prior work in multi-agent reinforcement learning and world models, including:

- Schwarting, W., Seyde, T., Gilitschenski, I., Liebenwein, L., Sander, R., Karaman, S., and Rus, D. (2020). *Deep Latent Competition: Learning to Race Using Visual Control Policies in Latent Space*. Conference on Robot Learning (CoRL 2020). https://arxiv.org/abs/2102.09812
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018. https://arxiv.org/abs/1801.01290
- Ha, D., and Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122. https://arxiv.org/abs/1803.10122

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.