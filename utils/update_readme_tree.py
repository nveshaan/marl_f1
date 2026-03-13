#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

CONFIGS_DIR = ROOT / "configs"
TRAIN_CONFIG = CONFIGS_DIR / "train.yaml"
MCR_DIR = ROOT / "multi_car_racing"

BEGIN = "<!-- BEGIN:PROJECT_TREE -->"
END = "<!-- END:PROJECT_TREE -->"

# Keep this aligned with what you want visible in README.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "experiments",
    "notebooks",
    ".vscode",
    ".ruff_cache",
}
EXCLUDE_SUFFIXES = {".pyc", ".log", ".yaml"}
EXCLUDE_EXACT = {"uv.lock", ".DS_Store", ".python-version", "update_readme_tree.py", "__init__.py"}
MAX_DEPTH = 3


def visible(path: Path) -> bool:
    if path == TRAIN_CONFIG:
        return True

    if path.name in EXCLUDE_EXACT:
        return False
    if path.is_dir() and path.name in EXCLUDE_DIRS:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False

    # Keep `configs/` itself and `configs/train.yaml`, hide everything else inside it.
    if path != CONFIGS_DIR and path.is_relative_to(CONFIGS_DIR):
        return False

    if path != MCR_DIR and path.is_relative_to(MCR_DIR):
        return False

    return not path.name.endswith(".egg-info")


def build_tree_lines(root: Path, max_depth: int) -> list[str]:
    lines = [root.name + "/"]

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return

        entries = [
            p
            for p in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            if visible(p)
        ]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(prefix + connector + name)

            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(entry, child_prefix, depth + 1)

    walk(root, "", 0)
    return lines


def update_readme(readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise RuntimeError(f"Missing markers in {readme_path}: {BEGIN} ... {END}")

    tree_block = "```text\n" + "\n".join(build_tree_lines(ROOT, MAX_DEPTH)) + "\n```"

    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    updated = text[:start] + "\n" + tree_block + "\n" + text[end:]

    if updated != text:
        readme_path.write_text(updated, encoding="utf-8")
        print("README tree updated.")
    else:
        print("README tree already up to date.")


if __name__ == "__main__":
    update_readme(README)
