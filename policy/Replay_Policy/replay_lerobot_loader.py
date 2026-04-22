"""Load LeRobot parquet replay trajectories and first-frame metadata."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to read LeRobot parquet replay data. "
            "Install it on the RoboTwin server with `pip install pyarrow`."
        ) from exc
    return pq


def load_dataset_info(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    info_path = data_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset info.json not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _episode_chunk(info: dict, episode_index: int) -> int:
    chunk_size = int(info.get("chunks_size", 1000))
    return episode_index // chunk_size


def resolve_episode_parquet_path(data_dir: str | Path, episode_index: int) -> Path:
    data_dir = Path(data_dir)
    info = load_dataset_info(data_dir)
    data_path_tmpl = info["data_path"]
    rel_path = data_path_tmpl.format(
        episode_chunk=_episode_chunk(info, episode_index),
        episode_index=episode_index,
    )
    parquet_path = data_dir / rel_path
    fallback = data_dir / "data" / f"episode_{episode_index:06d}.parquet"
    if parquet_path.is_file():
        return parquet_path
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Episode parquet not found for episode {episode_index}: tried {parquet_path} and {fallback}"
    )


def resolve_episode_video_path(
    data_dir: str | Path,
    episode_index: int,
    video_key: str = "observation.images.wrist",
) -> Path:
    data_dir = Path(data_dir)
    info = load_dataset_info(data_dir)
    rel_path = info["video_path"].format(
        episode_chunk=_episode_chunk(info, episode_index),
        episode_index=episode_index,
        video_key=video_key,
    )
    video_path = data_dir / rel_path
    if video_path.is_file():
        return video_path
    raise FileNotFoundError(
        f"Episode video not found for episode {episode_index}, video key {video_key}: {video_path}"
    )


def extract_first_frame(video_path: str | Path, output_path: str | Path) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        "select=eq(n\\,0)",
        "-vframes",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"ffmpeg did not create first-frame image: {output_path}")
    return output_path


def _load_column(path: Path, column: str):
    pq = _require_pyarrow()
    table = pq.read_table(path, columns=[column])
    return table[column].to_pylist()


def load_episode_state_sequence(
    data_dir: str | Path,
    episode_index: int,
    state_column: str = "observation.state",
    timestamp_column: str = "timestamp",
    gripper_index: int = 6,
) -> dict:
    parquet_path = resolve_episode_parquet_path(data_dir, episode_index)
    states = np.asarray(_load_column(parquet_path, state_column), dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 6:
        raise ValueError(
            f"Unexpected state shape from {state_column}: expected [T, >=6], got {states.shape}"
        )

    try:
        timestamps = np.asarray(_load_column(parquet_path, timestamp_column), dtype=np.float64).reshape(-1)
    except Exception:
        timestamps = np.arange(len(states), dtype=np.float64)

    gripper = states[:, gripper_index] if states.shape[1] > gripper_index else np.ones(len(states), dtype=np.float64)

    return {
        "poses_real": states[:, :6].copy(),
        "gripper": gripper.astype(np.float64),
        "times": timestamps,
        "length": len(states),
        "parquet_path": str(parquet_path),
    }


def load_first_frame_state(
    data_dir: str | Path,
    episode_index: int,
    state_column: str = "observation.state",
) -> np.ndarray:
    sequence = load_episode_state_sequence(
        data_dir=data_dir,
        episode_index=episode_index,
        state_column=state_column,
    )
    if sequence["length"] == 0:
        raise ValueError(f"Episode {episode_index} is empty.")
    return np.asarray(sequence["poses_real"][0], dtype=np.float64)
