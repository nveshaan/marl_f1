from pathlib import Path

import portalocker


def next_run_index(
    name: str, base_dir: str = "experiments", reserve: bool = True, lock_timeout: float = 10.0
) -> str:
    """Return next available numeric suffix for `name` under `base_dir`.

    Uses `portalocker` to acquire an exclusive file lock on a lockfile inside
    `base_dir`. If `reserve` is True the chosen directory is created (reserved)
    while the lock is held. Returns the numeric index as a string.
    """
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    lock_path = base / ".run_index.lock"
    # Open the lock file and acquire an exclusive lock with timeout
    with open(lock_path, "a+") as lockf:
        portalocker.lock(lockf, portalocker.LOCK_EX)
        try:
            max_idx = -1
            prefix = f"{name}_"
            for entry in base.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name == name:
                    max_idx = max(max_idx, 0)
                elif entry.name.startswith(prefix):
                    suffix = entry.name[len(prefix) :]
                    if suffix.isdigit():
                        idx = int(suffix)
                        if idx > max_idx:
                            max_idx = idx

            next_idx = max_idx + 1 if max_idx >= 0 else 0
            chosen = f"{name}_{next_idx}"
            if reserve:
                (base / chosen).mkdir(parents=True, exist_ok=True)
        finally:
            portalocker.unlock(lockf)

    return str(next_idx)
