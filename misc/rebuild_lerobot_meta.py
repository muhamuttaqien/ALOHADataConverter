#!/usr/bin/env python3

import argparse
import copy
import json
import re
import sys
import uuid
from pathlib import Path


def _extend_import_paths():
    search_roots = [
        Path(__file__).resolve().parents[1] / ".venv" / "lib",
        Path.home() / ".local" / "lib",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for site_packages in sorted(root.glob("python*/site-packages")):
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                sys.path.insert(0, site_packages_str)


try:
    import numpy as np
except ModuleNotFoundError:
    _extend_import_paths()
    import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    _extend_import_paths()
    import pyarrow.parquet as pq


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")
CHUNK_RE = re.compile(r"chunk-(\d+)$")
META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def parse_episode_index(path):
    match = EPISODE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer episode index from {path}")
    return int(match.group(1))


def infer_chunk_index(path):
    for parent in path.parents:
        match = CHUNK_RE.match(parent.name)
        if match:
            return int(match.group(1))
    return 0


def list_episode_parquet_files(dataset_dir):
    return sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))


def list_video_files(dataset_dir):
    return sorted((dataset_dir / "videos").glob("chunk-*/*/episode_*.mp4"))


def existing_meta_is_protected(meta_dir):
    for name in META_FILES:
        path = meta_dir / name
        if path.exists() and path.stat().st_size > 0:
            return True
    return False


def scalar_json(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def list_json(values):
    return [scalar_json(value) for value in values]


def values_to_array(values):
    converted = []
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.dtype.kind not in {"b", "i", "u", "f"}:
            continue
        converted.append(arr)

    if not converted:
        return None

    try:
        arr = np.stack(converted)
    except ValueError:
        return None

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        return None

    return arr


def compute_column_stats(column):
    arr = values_to_array(column.to_pylist())
    if arr is None or len(arr) == 0:
        return None

    numeric = arr.astype(np.float64)
    return {
        "count": [int(arr.shape[0])],
        "max": list_json(np.max(arr, axis=0).tolist()),
        "mean": list_json(np.mean(numeric, axis=0).tolist()),
        "min": list_json(np.min(arr, axis=0).tolist()),
        "std": list_json(np.std(numeric, axis=0).tolist()),
    }


def compute_episode_stats(parquet_path):
    table = pq.read_table(parquet_path)
    stats = {}
    for column_name in table.column_names:
        column_stats = compute_column_stats(table[column_name])
        if column_stats is not None:
            stats[column_name] = column_stats
    return stats


def read_unique_ints(table, column_name, default):
    if column_name not in table.column_names:
        return {default}
    values = table[column_name].to_pylist()
    return {int(value) for value in values if value is not None}


def collect_episodes(dataset_dir, default_task_index):
    episodes = []
    task_indices = set()

    for parquet_path in list_episode_parquet_files(dataset_dir):
        episode_index = parse_episode_index(parquet_path)
        parquet_file = pq.ParquetFile(parquet_path)
        schema_names = set(parquet_file.schema_arrow.names)
        columns = [column for column in ("episode_index", "task_index") if column in schema_names]
        table = pq.read_table(parquet_path, columns=columns)
        parquet_episode_indices = read_unique_ints(table, "episode_index", episode_index)
        parquet_task_indices = read_unique_ints(table, "task_index", default_task_index)
        task_indices.update(parquet_task_indices)
        episodes.append(
            {
                "episode_index": min(parquet_episode_indices),
                "file_episode_index": episode_index,
                "chunk_index": infer_chunk_index(parquet_path),
                "length": int(parquet_file.metadata.num_rows),
                "task_indices": sorted(parquet_task_indices),
                "parquet_path": parquet_path,
            }
        )

    episodes.sort(key=lambda row: row["episode_index"])
    return episodes, sorted(task_indices)


def make_task_rows(task_indices, task_name):
    rows = []
    for task_index in task_indices:
        rows.append({"task_index": int(task_index), "task": task_name})
    return rows


def make_episode_rows(dataset_dir, episodes, task_name):
    rows = []
    for episode in episodes:
        episode_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_dir}:{episode['episode_index']}"))
        rows.append(
            {
                "episode_index": int(episode["episode_index"]),
                "length": int(episode["length"]),
                "tasks": [task_name],
                "task_type": "SHT",
                "task_success": True,
                "short_horizon_task": [task_name],
                "success_short_horizon_task": True,
                "primitive_action": [],
                "uuid": episode_uuid,
                "metadata": {
                    "uuid": episode_uuid,
                    "reconstructed": True,
                    "source": "misc/rebuild_lerobot_meta.py",
                },
            }
        )
    return rows


def make_episode_stats_rows(episodes):
    rows = []
    for episode in episodes:
        rows.append(
            {
                "episode_index": int(episode["episode_index"]),
                "stats": compute_episode_stats(episode["parquet_path"]),
            }
        )
    return rows


def rebuild_info(template_info, dataset_dir, episodes, task_indices):
    info = copy.deepcopy(template_info)
    chunk_indices = {episode["chunk_index"] for episode in episodes}

    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(sum(episode["length"] for episode in episodes))
    info["total_tasks"] = len(task_indices)
    info["total_videos"] = len(list_video_files(dataset_dir))
    info["total_chunks"] = len(chunk_indices)
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info.setdefault("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    info.setdefault("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return info


def rebuild_meta(dataset_dir, template_dataset_dir, task_name, default_task_index, with_stats, force):
    dataset_dir = dataset_dir.expanduser().resolve()
    template_dataset_dir = template_dataset_dir.expanduser().resolve()
    meta_dir = dataset_dir / "meta"

    if not (dataset_dir / "data").is_dir():
        raise FileNotFoundError(f"Missing data directory: {dataset_dir / 'data'}")
    if not (template_dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing template info.json: {template_dataset_dir / 'meta' / 'info.json'}")
    if existing_meta_is_protected(meta_dir) and not force:
        raise RuntimeError(f"{meta_dir} contains non-empty meta files. Re-run with --force to overwrite.")

    template_info = read_json(template_dataset_dir / "meta" / "info.json")
    episodes, task_indices = collect_episodes(dataset_dir, default_task_index)
    if not episodes:
        raise RuntimeError(f"No Parquet episodes found under {dataset_dir / 'data'}")

    task_rows = make_task_rows(task_indices, task_name)
    episode_rows = make_episode_rows(dataset_dir, episodes, task_name)
    info = rebuild_info(template_info, dataset_dir, episodes, task_indices)

    meta_dir.mkdir(parents=True, exist_ok=True)
    write_json(meta_dir / "info.json", info)
    write_jsonl(meta_dir / "tasks.jsonl", task_rows)
    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    if with_stats:
        write_jsonl(meta_dir / "episodes_stats.jsonl", make_episode_stats_rows(episodes))
    else:
        write_jsonl(meta_dir / "episodes_stats.jsonl", [{"episode_index": int(row["episode_index"]), "stats": {}} for row in episode_rows])

    return {
        "dataset_dir": str(dataset_dir),
        "template_dataset_dir": str(template_dataset_dir),
        "episodes": len(episodes),
        "frames": info["total_frames"],
        "tasks": task_rows,
        "videos": info["total_videos"],
        "stats": with_stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild missing LeRobot meta files from Parquet episodes and a valid template info.json.")
    parser.add_argument("--dataset_dir", required=True, help="LeRobot dataset whose meta files should be rebuilt")
    parser.add_argument("--template_dataset_dir", required=True, help="Valid LeRobot dataset with matching feature schema")
    parser.add_argument("--task", default="task", help="Task label to use in tasks.jsonl and episodes.jsonl")
    parser.add_argument("--default_task_index", type=int, default=0, help="Task index used when Parquet has no task_index column")
    parser.add_argument("--with_stats", action="store_true", help="Compute numeric episodes_stats.jsonl from Parquet columns")
    parser.add_argument("--force", action="store_true", help="Overwrite non-empty meta files")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = rebuild_meta(
        dataset_dir=Path(args.dataset_dir),
        template_dataset_dir=Path(args.template_dataset_dir),
        task_name=args.task,
        default_task_index=args.default_task_index,
        with_stats=args.with_stats,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
