#!/usr/bin/env python3

import os
import cv2
import h5py
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import natsort

# Constants
VALIDITY_LABEL = "valid"

DEFAULT_CAMERAS = ["cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist"]
DCAMERA_NAMES = ["dcam_high", "dcam_low"]

def decode_hdf5(dataset_path):
    """Read one HDF5 episode file and return (is_sim, qpos, qvel, effort, action, image_dict)."""
    if not os.path.isfile(dataset_path):
        print(f"❌ Dataset does not exist at {dataset_path}")
        return None

    with h5py.File(dataset_path, "r") as root:
        is_sim = root.attrs.get("sim", False)
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        effort = root["/observations/effort"][()]
        action = root["/action"][()]

        image_dict = {}
        if "/observations/images" in root:
            for cam_name in root["/observations/images"].keys():
                emc_images = root[f"/observations/images/{cam_name}"][()]
                # Keep as-is (may be JPEG bytes or raw arrays); we decode later
                image_dict[cam_name] = [img for img in emc_images]

    return is_sim, qpos, qvel, effort, action, image_dict

def _to_2d(arr):
    """Ensure (T, D) shape."""
    arr = np.asarray(arr)
    return arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)

def process_dataset(task_folder: Path, output_dir: Path, cameras, fps: float):
    print(f"📦 Processing task: {task_folder.name}")

    INPUT_DIR = task_folder
    OUT_DIR   = output_dir / task_folder.name
    DATA_DIR  = OUT_DIR / "data/chunk-000"
    META_DIR  = OUT_DIR / "meta"
    VIDEO_DIR = OUT_DIR / "videos/chunk-000"

    # Make directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        (VIDEO_DIR / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

    # Episodes
    episode_files = natsort.natsorted(INPUT_DIR.glob("episode_*.hdf5"))
    if len(episode_files) == 0:
        print(f"⚠️ No episodes in {INPUT_DIR.resolve()}; skip.")
        return

    # Infer dims & per-camera shapes from first episode
    decoded = decode_hdf5(str(episode_files[0]))
    if decoded is None:
        print("⚠️ Failed to read first episode; skip task.")
        return
    first_is_sim, first_qpos, first_qvel, first_effort, first_action, first_images = decoded
    first_qpos, first_qvel, first_effort, first_action = map(_to_2d, [first_qpos, first_qvel, first_effort, first_action])

    QPOS_DIM   = first_qpos.shape[1]
    QVEL_DIM   = first_qvel.shape[1]
    EFFORT_DIM = first_effort.shape[1]
    ACTION_DIM = first_action.shape[1]
    STATE_DIM  = QPOS_DIM + QVEL_DIM + EFFORT_DIM

    video_shapes = {}
    for cam in cameras:
        frames = first_images.get(cam, [])
        if len(frames) > 0:
            sample = frames[0]
            if isinstance(sample, (bytes, bytearray)):
                # JPEG-encoded bytes → decode to BGR for shape
                sample = cv2.imdecode(np.frombuffer(sample, dtype=np.uint8), cv2.IMREAD_COLOR)
            # fallback if decode returns None (shouldn’t happen normally)
            if sample is None:
                video_shapes[cam] = (480, 640, 3)
            else:
                H, W = sample.shape[:2]
                video_shapes[cam] = (H, W, 3)
        else:
            video_shapes[cam] = (480, 640, 3)  # fallback

    print(
        f"dims: qpos={QPOS_DIM}, qvel={QVEL_DIM}, effort={EFFORT_DIM}, "
        f"action={ACTION_DIM}, state={STATE_DIM}"
    )

    # Per-task metadata
    episodes_meta = []
    unique_tasks = set()
    TASK_DESCRIPTION = task_folder.name.replace("_", " ")
    episode_count = 0

    # ---------- Episode loop ----------
    for ep_id, ep_path in enumerate(episode_files):
        episode_id = f"episode_{ep_id:06d}"
        print(f"\n📦 Processing {episode_id}.hdf5")

        decoded = decode_hdf5(str(ep_path))
        if decoded is None:
            print(f"⚠️ Cannot read {ep_path}; skip episode.")
            continue

        is_sim, qpos, qvel, effort, action, image_dict = decoded

        # (T, D) float32
        qpos   = _to_2d(qpos).astype(np.float32, copy=False)
        qvel   = _to_2d(qvel).astype(np.float32, copy=False)
        effort = _to_2d(effort).astype(np.float32, copy=False)
        action = _to_2d(action).astype(np.float32, copy=False)

        # Dim sanity
        if (
            qpos.shape[1]   != QPOS_DIM or
            qvel.shape[1]   != QVEL_DIM or
            effort.shape[1] != EFFORT_DIM or
            action.shape[1] != ACTION_DIM
        ):
            print(f"⚠️ Dim mismatch in {ep_path.name}; skip episode.")
            continue

        num_steps = action.shape[0]

        for cam in cameras:
            frames = image_dict.get(cam, [])
            if len(frames) == 0:
                continue

            h, w, _ = video_shapes[cam]
            video_path = VIDEO_DIR / f"observation.images.{cam}" / f"{episode_id}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            print(f"🎞️ Saving video: {video_path}")

            for img in frames:
                if isinstance(img, (bytes, bytearray)):
                    # Decode to BGR
                    img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if img.shape[0] != h or img.shape[1] != w:
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    writer.write(img)
                else:
                    # Assume RGB ndarray → convert to BGR for writing
                    if img is None:
                        continue
                    if img.shape[0] != h or img.shape[1] != w:
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            writer.release()

        # --- Save parquet with Arrow vector columns
        timestamps = (np.arange(num_steps, dtype=np.float64) / fps)
        next_done = np.zeros((num_steps,), dtype=bool)
        next_done[-1] = True

        state = np.concatenate([qpos, qvel, effort], axis=1)  # (T, STATE_DIM)

        table = pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.ravel(order="C"), type=pa.float32()), STATE_DIM
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.ravel(order="C"), type=pa.float32()), ACTION_DIM
            ),
            "timestamp": pa.array(timestamps),
            "frame_index": pa.array(np.arange(num_steps, dtype=np.int64)),
            "episode_index": pa.array(np.full((num_steps,), ep_id, dtype=np.int64)),
            "index": pa.array(np.arange(ep_id * 10000, ep_id * 10000 + num_steps, dtype=np.int64)),
            "task_index": pa.array(np.zeros((num_steps,), dtype=np.int64)),
            "next.done": pa.array(next_done),
        })
        pq.write_table(table, DATA_DIR / f"{episode_id}.parquet")

        # --- Episode metadata
        episodes_meta.append({
            "episode_index": ep_id,
            "tasks": [TASK_DESCRIPTION, VALIDITY_LABEL],
            "length": int(num_steps),
        })
        unique_tasks.add((TASK_DESCRIPTION, VALIDITY_LABEL))
        episode_count += 1

    # ---------- Write meta files ----------
    with open(META_DIR / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(META_DIR / "tasks.jsonl", "w") as f:
        for task_index, (task_description, _) in enumerate(sorted(unique_tasks)):
            f.write(json.dumps({"task_index": task_index, "task": task_description}) + "\n")

    # modality.json
    modality_config = {
        "state": {
            "qpos":   {"start": 0, "end": QPOS_DIM},
            "qvel":   {"start": QPOS_DIM, "end": QPOS_DIM + QVEL_DIM},
            "effort": {"start": QPOS_DIM + QVEL_DIM, "end": QPOS_DIM + QVEL_DIM + EFFORT_DIM},
        },
        "action": {"qpos": {"start": 0, "end": ACTION_DIM}},
        "video": {},
        "annotation": {
            "human": {
                "action": {
                    "task_description": {"type": "text"},
                    "validity": {"type": "text"},
                }
            }
        }
    }
    for cam in cameras:
        modality_config["video"][cam] = {"original_key": f"observation.images.{cam}"}

    with open(META_DIR / "modality.json", "w") as f:
        json.dump(modality_config, f, indent=2)

    # info.json
    total_frames = 0
    parq_files = natsort.natsorted(DATA_DIR.glob("episode_*.parquet"))
    for pf in parq_files:
        try:
            total_frames += len(pd.read_parquet(pf))
        except Exception as e:
            print(f"⚠️ Skipping parquet '{pf.name}': {e}")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": [f"state_{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [f"motor_{i}" for i in range(ACTION_DIM)],
        },
        "timestamp":    {"dtype": "float64", "shape": [1]},
        "task_index":   {"dtype": "int64",   "shape": [1]},
        "episode_index":{"dtype": "int64",   "shape": [1]},
        "index":        {"dtype": "int64",   "shape": [1]},
        "next.done":    {"dtype": "bool",    "shape": [1]},
    }
    for cam in cameras:
        H, W, C = video_shapes.get(cam, (480, 640, 3))
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [H, W, C],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "mp4v",
                "video.pix_fmt": "bgr24",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v1.0",
        "robot_type": "ALOHA",
        "total_episodes": len(parq_files),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": len(parq_files),  # one set of per-camera videos per episode
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(parq_files)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(META_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # stats.json
    print("📊 Generating stats.json ...")
    all_parquet_files = natsort.natsorted(DATA_DIR.glob("episode_*.parquet"))
    if len(all_parquet_files) == 0:
        print("⚠️ No parquet files; skip stats.")
        return

    all_dfs = []
    for i, fpath in enumerate(all_parquet_files):
        try:
            df = pd.read_parquet(fpath)
            df["task_index"] = i  # keep your convention
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skip '{fpath.name}' while building stats: {e}")

    if len(all_dfs) == 0:
        print("⚠️ No data for stats.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    stats = {}
    for col in df_all.columns:
        try:
            values = np.vstack(df_all[col].values).astype(np.float32)
            stats[col] = {
                "mean": np.mean(values, axis=0).tolist(),
                "std":  np.std(values,  axis=0).tolist(),
                "min":  np.min(values,  axis=0).tolist(),
                "max":  np.max(values,  axis=0).tolist(),
                "q01":  np.quantile(values, 0.01, axis=0).tolist(),
                "q99":  np.quantile(values, 0.99, axis=0).tolist(),
            }
        except Exception as e:
            # non-vector columns (e.g., bool) or empty — just skip politely
            print(f"⚠️ Skipping column '{col}' in stats: {e}")

    with open(META_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Done: {task_folder.name} → {len(parq_files)} episodes.\n")

def main():
    parser = argparse.ArgumentParser(description="Convert ALOHA HDF5 dataset to GR00T format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input HDF5 dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for the GR00T format.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (fps).")
    parser.add_argument("--cameras", type=str, nargs="*", default=DEFAULT_CAMERAS, help="Camera names to export (default: cam_high cam_left_wrist cam_low cam_right_wrist).",)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate task folders
    for dataset_folder in natsort.natsorted([p for p in input_dir.iterdir() if p.is_dir()]):
        process_dataset(Path(dataset_folder), output_dir, args.cameras, args.fps)

if __name__ == "__main__":
    main()
