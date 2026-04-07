#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _extend_import_paths():
    search_roots = [
        Path(__file__).resolve().parent / ".venv" / "lib",
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
    import cv2
except ModuleNotFoundError:
    _extend_import_paths()
    import cv2

try:
    import h5py
except ModuleNotFoundError:
    _extend_import_paths()
    import h5py

import numpy as np

try:
    import natsort
except ModuleNotFoundError:
    natsort = None


BUILDER_TEMPLATE = """from typing import Any, Iterator, Tuple

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def build_sample_indices(length, source_fps, target_fps):
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)

    if target_fps <= 0 or source_fps <= 0 or np.isclose(source_fps, target_fps):
        return np.arange(length, dtype=np.int64)

    duration = length / source_fps
    target_len = max(1, int(round(duration * target_fps)))
    sample_times = np.arange(target_len, dtype=np.float64) / target_fps
    indices = np.round(sample_times * source_fps).astype(np.int64)
    return np.clip(indices, 0, length - 1)


def extract_task_desc(root):
    if "/text/prompt" in root:
        prompt = root["/text/prompt"][()]
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
        return str(prompt)

    for attr_name in ("task_desc", "Task name", "task_name", "task"):
        if attr_name in root.attrs:
            value = root.attrs[attr_name]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

    return "{dummy_instruction}"


def infer_source_fps(root, default_fps):
    for attr_name in ("frame_rate", "fps"):
        if attr_name in root.attrs:
            return float(root.attrs[attr_name])
    return float(default_fps)


def decode_rgb_frame(encoded):
    frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode RGB frame")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def convert_depth_frames(depth_frames):
    if np.issubdtype(depth_frames.dtype, np.floating):
        return np.clip(np.round(depth_frames * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if depth_frames.dtype == np.uint16:
        return depth_frames
    if np.issubdtype(depth_frames.dtype, np.integer):
        return depth_frames.astype(np.uint16)
    raise TypeError(f"Unsupported depth dtype: {{depth_frames.dtype}}")


class {class_name}(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {{
        "1.0.0": "Initial release.",
    }}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._manifest_path = Path(__file__).resolve().parent / "{manifest_name}"
        with self._manifest_path.open("r", encoding="utf-8") as f:
            self._manifest = json.load(f)

    def _info(self) -> tfds.core.DatasetInfo:
        observation_features = {{
{observation_feature_lines}
        }}

        step_features = {{
            "observation": tfds.features.FeaturesDict(observation_features),
            "action": tfds.features.Tensor(
                shape=({action_dim},),
                dtype=np.float32,
                doc="Robot action vector from the source ALOHA HDF5.",
            ),
            "discount": tfds.features.Scalar(
                dtype=np.float32,
                doc="Discount factor. Demos use 1.0 for all steps.",
            ),
            "reward": tfds.features.Scalar(
                dtype=np.float32,
                doc="Reward signal. Demos use 1.0 on the last step and 0.0 otherwise.",
            ),
            "is_first": tfds.features.Scalar(dtype=np.bool_),
            "is_last": tfds.features.Scalar(dtype=np.bool_),
            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
            "language_instruction": tfds.features.Text(
                doc="Language instruction extracted from the source HDF5 metadata.",
            ),
            "language_embedding": tfds.features.Tensor(
                shape=({language_embedding_dim},),
                dtype=np.float32,
                doc="Language embedding from /text/text_embedding when available.",
            ),
            "timestamp": tfds.features.Scalar(
                dtype=np.float32,
                doc="Step timestamp in seconds after optional FPS resampling.",
            ),
            "source_index": tfds.features.Scalar(
                dtype=np.int32,
                doc="Original source frame index before resampling.",
            ),
        }}

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({{
                "steps": tfds.features.Dataset(step_features),
                "episode_metadata": tfds.features.FeaturesDict({{
                    "file_path": tfds.features.Text(doc="Path to the original HDF5 episode."),
                    "task_name": tfds.features.Text(doc="Dataset folder / task name."),
                    "episode_length": tfds.features.Scalar(dtype=np.int32),
                    "source_fps": tfds.features.Scalar(dtype=np.float32),
                }}),
            }})
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {{
{split_generator_lines}
        }}

    def _generate_examples(self, split_name) -> Iterator[Tuple[str, Any]]:
        episode_paths = self._manifest["splits"][split_name]
        target_fps = float(self._manifest["target_fps"])
        language_embedding_dim = int(self._manifest["language_embedding_dim"])

        for episode_path in episode_paths:
            with h5py.File(episode_path, "r") as root:
                qpos = root["/observations/qpos"][()].astype(np.float32)
                qvel = root["/observations/qvel"][()].astype(np.float32)
                effort = root["/observations/effort"][()].astype(np.float32)
                action = root["/action"][()].astype(np.float32)
                state = np.concatenate([qpos, qvel, effort], axis=1)

                source_fps = infer_source_fps(root, target_fps)
                sample_indices = build_sample_indices(len(action), source_fps, target_fps)
                task_desc = extract_task_desc(root)

                if "/text/text_embedding" in root:
                    language_embedding = root["/text/text_embedding"][()].astype(np.float32)
                else:
                    language_embedding = np.zeros((language_embedding_dim,), dtype=np.float32)

                rgb_cache = {{}}
                for cam_name in self._manifest["rgb_cameras"]:
                    dataset = root[f"/observations/images/{{cam_name}}"]
                    rgb_cache[cam_name] = [decode_rgb_frame(dataset[idx]) for idx in sample_indices]

                depth_cache = {{}}
                for cam_name in self._manifest["depth_cameras"]:
                    dataset = root[f"/observations/depth/{{cam_name}}"]
                    depth_cache[cam_name] = convert_depth_frames(dataset[sample_indices])[..., None]

                steps = []
                for step_idx, src_idx in enumerate(sample_indices):
                    observation = {{
                        "state": state[src_idx].astype(np.float32),
                        "qpos": qpos[src_idx].astype(np.float32),
                        "qvel": qvel[src_idx].astype(np.float32),
                        "effort": effort[src_idx].astype(np.float32),
                    }}

                    for cam_name in self._manifest["rgb_cameras"]:
                        observation[f"{{cam_name}}_rgb"] = rgb_cache[cam_name][step_idx]

                    for cam_name in self._manifest["depth_cameras"]:
                        observation[f"{{cam_name}}_depth"] = depth_cache[cam_name][step_idx]

                    is_last = step_idx == (len(sample_indices) - 1)
                    steps.append({{
                        "observation": observation,
                        "action": action[src_idx].astype(np.float32),
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last else 0.0),
                        "is_first": step_idx == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": task_desc,
                        "language_embedding": language_embedding,
                        "timestamp": np.float32(src_idx / source_fps),
                        "source_index": np.int32(src_idx),
                    }})

                yield episode_path, {{
                    "steps": steps,
                    "episode_metadata": {{
                        "file_path": episode_path,
                        "task_name": self._manifest["dataset_name"],
                        "episode_length": len(steps),
                        "source_fps": np.float32(source_fps),
                    }},
                }}
"""


README_TEMPLATE = """# {dataset_name}

RLDS dataset builder generated from ALOHA HDF5 episodes.

- Source format: HDF5 episodes
- Generated by: `convert_to_rlds.py`
- Target format: TFDS / RLDS builder package compatible with the layout used in `third_party/rlds_dataset_builder`

## Build

Run this from inside this directory in an environment that has `tensorflow` and `tensorflow_datasets` installed:

```bash
tfds build --overwrite
```
"""


SETUP_TEMPLATE = """from setuptools import setup

setup(name="{dataset_name}", packages=["{dataset_name}"])
"""


def natsorted_paths(paths):
    if natsort is not None:
        return list(natsort.natsorted(paths))
    return sorted(paths, key=lambda path: path.name)


def safe_dataset_name(name):
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "aloha_rlds_dataset"
    if value[0].isdigit():
        value = f"dataset_{value}"
    return value


def camel_case(name):
    return "".join(part.capitalize() for part in name.split("_"))


def extract_task_desc(root):
    if "/text/prompt" in root:
        prompt = root["/text/prompt"][()]
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
        return str(prompt)

    for attr_name in ("task_desc", "Task name", "task_name", "task"):
        if attr_name in root.attrs:
            value = root.attrs[attr_name]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

    return "perform the demonstrated task"


def infer_source_fps(root, default_fps):
    for attr_name in ("frame_rate", "fps"):
        if attr_name in root.attrs:
            return float(root.attrs[attr_name])
    return float(default_fps)


def build_sample_indices(length, source_fps, target_fps):
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)

    if target_fps <= 0 or source_fps <= 0 or np.isclose(source_fps, target_fps):
        return np.arange(length, dtype=np.int64)

    duration = length / source_fps
    target_len = max(1, int(round(duration * target_fps)))
    sample_times = np.arange(target_len, dtype=np.float64) / target_fps
    indices = np.round(sample_times * source_fps).astype(np.int64)
    return np.clip(indices, 0, length - 1)


def iter_dataset_folders(input_path):
    input_path = Path(input_path)

    if input_path.is_file() and input_path.suffix == ".hdf5":
        return [(input_path.parent.name, [input_path])]

    if input_path.is_dir():
        direct_files = natsorted_paths(input_path.glob("episode*.hdf5"))
        if direct_files:
            return [(input_path.name, direct_files)]

        dataset_folders = []
        for dataset_folder in sorted(input_path.iterdir()):
            if not dataset_folder.is_dir():
                continue
            hdf5_files = natsorted_paths(dataset_folder.glob("episode*.hdf5"))
            if hdf5_files:
                dataset_folders.append((dataset_folder.name, hdf5_files))
        return dataset_folders

    return []


def infer_rgb_camera_shape(root, cam_name):
    dataset = root[f"/observations/images/{cam_name}"]
    frame = cv2.imdecode(dataset[0], cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to decode first RGB frame for {cam_name}")
    height, width = frame.shape[:2]
    return [height, width, 3]


def infer_depth_camera_shape(root, cam_name):
    dataset = root[f"/observations/depth/{cam_name}"]
    height, width = dataset.shape[1:3]
    return [height, width, 1]


def make_split_map(hdf5_files, val_ratio):
    if not hdf5_files:
        return {"train": []}

    if val_ratio <= 0.0:
        return {"train": [str(path.resolve()) for path in hdf5_files]}

    val_count = max(1, int(round(len(hdf5_files) * val_ratio)))
    val_count = min(val_count, len(hdf5_files) - 1) if len(hdf5_files) > 1 else 0

    train_files = hdf5_files[:-val_count] if val_count > 0 else hdf5_files
    val_files = hdf5_files[-val_count:] if val_count > 0 else []

    split_map = {"train": [str(path.resolve()) for path in train_files]}
    if val_files:
        split_map["val"] = [str(path.resolve()) for path in val_files]
    return split_map


def infer_dataset_manifest(dataset_name, hdf5_files, target_fps, val_ratio):
    first_file = hdf5_files[0]
    with h5py.File(first_file, "r") as root:
        qpos_dim = int(root["/observations/qpos"].shape[1])
        qvel_dim = int(root["/observations/qvel"].shape[1])
        effort_dim = int(root["/observations/effort"].shape[1])
        action_dim = int(root["/action"].shape[1])
        num_source_steps = int(root["/action"].shape[0])

        rgb_cameras = list(root["/observations/images"].keys()) if "/observations/images" in root else []
        depth_cameras = list(root["/observations/depth"].keys()) if "/observations/depth" in root else []

        rgb_shapes = {cam_name: infer_rgb_camera_shape(root, cam_name) for cam_name in rgb_cameras}
        depth_shapes = {cam_name: infer_depth_camera_shape(root, cam_name) for cam_name in depth_cameras}

        language_embedding_dim = int(root["/text/text_embedding"].shape[0]) if "/text/text_embedding" in root else 384
        task_desc = extract_task_desc(root)
        source_fps = infer_source_fps(root, target_fps)
        sample_indices = build_sample_indices(num_source_steps, source_fps, target_fps)

    return {
        "dataset_name": dataset_name,
        "target_fps": float(target_fps),
        "dummy_instruction": task_desc,
        "qpos_dim": qpos_dim,
        "qvel_dim": qvel_dim,
        "effort_dim": effort_dim,
        "action_dim": action_dim,
        "state_dim": int(qpos_dim + qvel_dim + effort_dim),
        "language_embedding_dim": int(language_embedding_dim),
        "rgb_cameras": rgb_cameras,
        "depth_cameras": depth_cameras,
        "rgb_shapes": rgb_shapes,
        "depth_shapes": depth_shapes,
        "num_steps_example": int(len(sample_indices)),
        "splits": make_split_map(hdf5_files, val_ratio),
    }


def build_observation_feature_lines(manifest):
    lines = [
        f'            "state": tfds.features.Tensor(shape=({manifest["state_dim"]},), dtype=np.float32, doc="Concatenated qpos, qvel and effort."),',
        f'            "qpos": tfds.features.Tensor(shape=({manifest["qpos_dim"]},), dtype=np.float32, doc="Measured joint positions."),',
        f'            "qvel": tfds.features.Tensor(shape=({manifest["qvel_dim"]},), dtype=np.float32, doc="Measured joint velocities."),',
        f'            "effort": tfds.features.Tensor(shape=({manifest["effort_dim"]},), dtype=np.float32, doc="Measured joint efforts."),',
    ]

    for cam_name in manifest["rgb_cameras"]:
        h, w, c = manifest["rgb_shapes"][cam_name]
        lines.append(
            f'            "{cam_name}_rgb": tfds.features.Image(shape=({h}, {w}, {c}), dtype=np.uint8, encoding_format="png", doc="RGB observation from {cam_name}."),'
        )

    for cam_name in manifest["depth_cameras"]:
        h, w, c = manifest["depth_shapes"][cam_name]
        lines.append(
            f'            "{cam_name}_depth": tfds.features.Tensor(shape=({h}, {w}, {c}), dtype=np.uint16, doc="Depth observation from {cam_name} stored in millimeter-like uint16 units."),'
        )

    return "\n".join(lines)


def build_split_generator_lines(manifest):
    lines = []
    for split_name in manifest["splits"].keys():
        lines.append(f'            "{split_name}": self._generate_examples(split_name="{split_name}"),')
    return "\n".join(lines)


def write_dataset_package(package_dir, manifest):
    package_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = manifest["dataset_name"]
    builder_class_name = camel_case(dataset_name)
    manifest_name = "dataset_manifest.json"
    builder_filename = f"{dataset_name}_dataset_builder.py"

    builder_source = BUILDER_TEMPLATE.format(
        class_name=builder_class_name,
        manifest_name=manifest_name,
        observation_feature_lines=build_observation_feature_lines(manifest),
        split_generator_lines=build_split_generator_lines(manifest),
        action_dim=manifest["action_dim"],
        language_embedding_dim=manifest["language_embedding_dim"],
        dummy_instruction=manifest["dummy_instruction"].replace("{", "{{").replace("}", "}}"),
    )

    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / builder_filename).write_text(builder_source, encoding="utf-8")
    (package_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (package_dir / "README.md").write_text(README_TEMPLATE.format(dataset_name=dataset_name), encoding="utf-8")
    (package_dir / "CITATIONS.bib").write_text("% Add dataset citation here.\n", encoding="utf-8")
    (package_dir / "setup.py").write_text(SETUP_TEMPLATE.format(dataset_name=dataset_name), encoding="utf-8")

    return package_dir / builder_filename


def maybe_build_tfds(package_dir, overwrite):
    tfds_cmd = shutil.which("tfds")
    if tfds_cmd is not None:
        cmd = [tfds_cmd, "build"]
    else:
        try:
            import tensorflow_datasets  # noqa: F401
        except ModuleNotFoundError:
            print(
                "⚠️  `tensorflow_datasets` is not installed in the current environment. "
                "Builder package was generated, but tfds build was skipped.\n"
                "   Install RLDS build dependencies first, for example:\n"
                "   uv pip install -e '.[rlds]'"
            )
            return
        cmd = [sys.executable, "-m", "tensorflow_datasets.scripts.cli.main", "build"]

    if overwrite:
        cmd.append("--overwrite")

    print(f"🏗️ Building TFDS package in {package_dir}")
    subprocess.run(cmd, cwd=package_dir, check=True)


def process_dataset_folder(dataset_name, hdf5_files, output_dir, fps, val_ratio, build_tfds, overwrite):
    safe_name = safe_dataset_name(dataset_name)
    package_dir = Path(output_dir) / safe_name
    print(f"\n📦 Processing folder: {dataset_name} -> {safe_name}")

    manifest = infer_dataset_manifest(safe_name, hdf5_files, fps, val_ratio)
    builder_path = write_dataset_package(package_dir, manifest)
    print(f"📝 Generated builder: {builder_path}")

    if build_tfds:
        maybe_build_tfds(package_dir, overwrite)


def main():
    parser = argparse.ArgumentParser(description="Generate RLDS / TFDS dataset builders from ALOHA HDF5 episodes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to an episode file, a dataset folder, or a root folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder for generated RLDS dataset packages.")
    parser.add_argument("--fps", type=float, default=25, help="Target FPS after optional resampling.")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of episodes reserved for val split.")
    parser.add_argument("--build_tfds", action="store_true", help="Run `tfds build` after generating each dataset package.")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to tfds build.")
    args = parser.parse_args()

    dataset_folders = iter_dataset_folders(args.input_dir)
    if not dataset_folders:
        print(f"❌ No HDF5 episodes found under: {args.input_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, hdf5_files in dataset_folders:
        process_dataset_folder(
            dataset_name=dataset_name,
            hdf5_files=hdf5_files,
            output_dir=output_dir,
            fps=args.fps,
            val_ratio=args.val_ratio,
            build_tfds=args.build_tfds,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
