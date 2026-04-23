"""Path resolution helpers for auto-init scripts."""

from __future__ import annotations

from pathlib import Path


AUTO_INIT_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = AUTO_INIT_DIR.parent
POLICY_DIR = REPLAY_POLICY_DIR.parent
REPO_ROOT = POLICY_DIR.parent


def resolve_cli_path(path: str | Path, fallback_base: str | Path | None = None) -> Path:
    """Resolve a user-provided CLI path.

    Relative paths are interpreted from the current working directory. If that
    candidate does not exist and ``fallback_base`` is provided, fall back to the
    supplied base directory.
    """

    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or fallback_base is None:
        return cwd_candidate
    return (Path(fallback_base) / path).resolve()


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a config path that is defined relative to the repository root."""

    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_existing_path(path: str | Path, *bases: str | Path) -> Path:
    """Resolve a relative path against several bases, preferring the first existing match."""

    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [(Path.cwd() / path).resolve()]
    candidates.extend((Path(base) / path).resolve() for base in bases)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if bases:
        return (Path(bases[0]) / path).resolve()
    return candidates[0]
