#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
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
    import h5py
except ModuleNotFoundError:
    _extend_import_paths()
    import h5py

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    _extend_import_paths()
    import pyarrow.parquet as pq

try:
    import videoio
except ModuleNotFoundError:
    _extend_import_paths()
    import videoio

try:
    import natsort
except ModuleNotFoundError:
    natsort = None

from convert_to_rmb import (
    UrdfKinematics,
    build_sample_indices,
    compute_eef_pose_rel,
    convert_depth_frames,
    previous_step_delta,
    quaternion_xyzw_from_rotation,
    resolve_robot_urdf,
)


DEFAULT_OUTPUT_FPS = 25.0
DEFAULT_ARM_JOINT_DIMS = (7, 7)
DEFAULT_ROBOT_NAME = "Aloha"
GRIPPER_NAME_PATTERN = re.compile(r"(gripper|finger|jaw|grip|claw|hand)", re.IGNORECASE)
KNOWN_ROBOT_LAYOUTS = {
    "aloha": {
        "arm_joint_dims": (7, 7),
        "gripper_indices": (6, 13),
        "arm_fk_dims": (6, 6),
    },
    "vx300s": {
        "arm_joint_dims": (7,),
        "gripper_indices": (6,),
        "arm_fk_dims": (6,),
    },
    "wx250s": {
        "arm_joint_dims": (7,),
        "gripper_indices": (6,),
        "arm_fk_dims": (6,),
    },
    "ws250s": {
        "arm_joint_dims": (7,),
        "gripper_indices": (6,),
        "arm_fk_dims": (6,),
    },
}


@dataclass(frozen=True)
class RobotLayout:
    arm_joint_dims: tuple[int, ...]
    gripper_indices: tuple[int, ...]
    arm_fk_dims: tuple[int, ...]
    arm_slices: tuple[slice, ...]
    eef_target_links: tuple[str | None, ...]
    robot_name: str

    @property
    def total_arm_joint_dim(self):
        return self.arm_slices[-1].stop if self.arm_slices else 0


@dataclass(frozen=True)
class VectorLayout:
    qpos_slice: slice | None
    qvel_slice: slice | None
    effort_slice: slice | None
    action_slice: slice | None


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    dataset_name: str
    info: dict
    episodes_meta: dict
    tasks_by_index: dict
    vector_layout: VectorLayout
    robot_layout: RobotLayout
    parquet_files: tuple[Path, ...]
    parquet_columns: tuple[tuple[str, str], ...]
    video_specs: tuple[dict, ...]


@dataclass(frozen=True)
class EpisodeJob:
    dataset_dir: Path
    dataset_name: str
    parquet_path: Path
    out_dir: Path
    info: dict
    episodes_meta: dict
    tasks_by_index: dict
    vector_layout: VectorLayout
    requested_fps: float | None
    camera_workers: int
    video_preset: str
    robot_urdf: str
    robot_layout: RobotLayout
    video_specs: tuple[dict, ...]


def natsorted_paths(paths):
    if natsort is not None:
        return list(natsort.natsorted(paths))
    return sorted(paths, key=lambda path: path.name)


def discover_lerobot_datasets(input_path):
    input_path = Path(input_path).expanduser().resolve()
    datasets = []

    if input_path.is_dir() and (input_path / "meta" / "info.json").is_file() and (input_path / "data").is_dir():
        return [input_path]

    if input_path.is_dir():
        for child in sorted(input_path.iterdir()):
            if child.is_dir() and (child / "meta" / "info.json").is_file() and (child / "data").is_dir():
                datasets.append(child.resolve())

    return datasets


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_dataset_metadata(dataset_dir):
    meta_dir = dataset_dir / "meta"
    info = read_json(meta_dir / "info.json")
    modality = read_json(meta_dir / "modality.json") if (meta_dir / "modality.json").exists() else {}

    episodes_meta = {}
    for row in read_jsonl(meta_dir / "episodes.jsonl"):
        episode_index = row.get("episode_index")
        if episode_index is None:
            continue
        episodes_meta[int(episode_index)] = row

    tasks_by_index = {}
    for row in read_jsonl(meta_dir / "tasks.jsonl"):
        task_index = row.get("task_index")
        task = row.get("task")
        if task_index is None or task is None:
            continue
        tasks_by_index[int(task_index)] = str(task)

    return info, episodes_meta, tasks_by_index, modality


def list_episode_parquet_files(dataset_dir):
    parquet_files = natsorted_paths((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if parquet_files:
        return parquet_files
    return natsorted_paths((dataset_dir / "data").rglob("*.parquet"))


def infer_episode_index(parquet_path):
    match = re.search(r"episode_(\d+)", parquet_path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer episode index from file name: {parquet_path.name}")


def infer_chunk_index(parquet_path):
    match = re.search(r"chunk-(\d+)", parquet_path.parent.name)
    if match:
        return int(match.group(1))
    return 0


def infer_fps_from_timestamps(timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return None

    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None

    return float(1.0 / np.median(deltas))


def resolve_source_fps(info, timestamps):
    if "fps" in info and info["fps"] is not None:
        return float(info["fps"])

    inferred = infer_fps_from_timestamps(timestamps)
    if inferred is not None:
        return inferred

    return DEFAULT_OUTPUT_FPS


def resolve_target_fps(source_fps, requested_fps):
    if requested_fps is None:
        return float(source_fps)
    return float(requested_fps)


def list_video_specs(info):
    specs = []
    features = info.get("features", {})
    if not isinstance(features, dict):
        return specs

    for feature_key, feature_spec in features.items():
        if not isinstance(feature_spec, dict):
            continue
        if feature_spec.get("dtype") != "video":
            continue

        video_info = feature_spec.get("video_info") or feature_spec.get("info") or {}
        camera_name = feature_key.split(".")[-1]
        specs.append(
            {
                "feature_key": feature_key,
                "camera_name": camera_name,
                "is_depth": bool(video_info.get("video.is_depth_map", False)),
            }
        )

    return specs


def resolve_episode_task_desc(dataset_name, episode_index, episodes_meta, tasks_by_index, task_indices):
    episode_meta = episodes_meta.get(episode_index, {})
    tasks = episode_meta.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])

    if task_indices is not None and len(task_indices) > 0:
        for task_index in task_indices:
            task_name = tasks_by_index.get(int(task_index))
            if task_name:
                return task_name

    if len(tasks_by_index) == 1:
        return next(iter(tasks_by_index.values()))

    return dataset_name.replace("_", " ")


def stack_list_column(column, dtype=np.float64):
    values = []
    for item in column.to_pylist():
        values.append(np.asarray(item, dtype=dtype))

    if not values:
        return np.zeros((0, 0), dtype=dtype)

    return np.stack(values, axis=0)


def scalar_column_to_numpy(column, dtype=np.float64):
    values = column.to_pylist()
    return np.asarray(values, dtype=dtype)


def resolve_column_name(schema_names, exact, prefix):
    if exact in schema_names:
        return exact

    candidates = [name for name in schema_names if name.startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise KeyError(f"Missing required column: '{exact}' or prefix '{prefix}*'")

    raise KeyError(
        f"Ambiguous columns for '{exact}'. Candidates: {', '.join(candidates)}"
    )


def string_array(values):
    dtype = h5py.string_dtype(encoding="utf-8")
    return np.asarray([str(value) for value in values], dtype=dtype)


def normalize_robot_type(robot_type):
    return re.sub(r"[^a-z0-9]+", "", str(robot_type or "").lower())


def info_features(info):
    features = info.get("features", {})
    if isinstance(features, dict):
        return features
    return {}


def get_feature_spec(info, feature_key):
    feature_spec = info_features(info).get(feature_key)
    if isinstance(feature_spec, dict):
        return feature_spec
    return None


def get_feature_shape_dim(info, feature_key):
    feature_spec = get_feature_spec(info, feature_key)
    if not feature_spec:
        return None

    shape = feature_spec.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def get_feature_names(info, feature_key):
    feature_spec = get_feature_spec(info, feature_key)
    if not feature_spec:
        return None

    names = feature_spec.get("names")
    if isinstance(names, list):
        normalized = tuple(str(name) for name in names)
        if len(normalized) > 1:
            return normalized

    field_descriptions = feature_spec.get("field_descriptions")
    feature_dim = get_feature_shape_dim(info, feature_key)
    if isinstance(field_descriptions, dict) and feature_dim is not None and feature_dim > 0:
        resolved = [None] * feature_dim
        for field_key, field_spec in field_descriptions.items():
            if not isinstance(field_spec, dict):
                continue
            indices = field_spec.get("indices")
            if not isinstance(indices, list) or len(indices) != 1:
                continue
            try:
                index = int(indices[0])
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= feature_dim:
                continue
            description = field_spec.get("description")
            if description:
                resolved[index] = str(description)
            else:
                resolved[index] = str(field_key).split("/")[-1]
        if all(name is not None for name in resolved):
            return tuple(resolved)

    if isinstance(names, list):
        return tuple(str(name) for name in names)
    return None


def parse_range_slice(spec):
    if not isinstance(spec, dict):
        return None
    if "start" not in spec or "end" not in spec:
        return None
    try:
        start = int(spec["start"])
        end = int(spec["end"])
    except (TypeError, ValueError):
        return None
    if start < 0 or end < start:
        return None
    return slice(start, end)


def slice_length(slice_obj):
    if slice_obj is None or slice_obj.stop is None or slice_obj.start is None:
        return None
    return max(0, int(slice_obj.stop) - int(slice_obj.start))


def resolve_vector_layout(info, modality):
    state_cfg = modality.get("state") if isinstance(modality, dict) else None
    action_cfg = modality.get("action") if isinstance(modality, dict) else None

    return VectorLayout(
        qpos_slice=parse_range_slice(state_cfg.get("qpos")) if isinstance(state_cfg, dict) else None,
        qvel_slice=parse_range_slice(state_cfg.get("qvel")) if isinstance(state_cfg, dict) else None,
        effort_slice=parse_range_slice(state_cfg.get("effort")) if isinstance(state_cfg, dict) else None,
        action_slice=parse_range_slice(action_cfg.get("qpos")) if isinstance(action_cfg, dict) else None,
    )


def slice_feature_names(names, slice_obj):
    if names is None or slice_obj is None:
        return None
    if slice_obj.stop is None or slice_obj.stop > len(names):
        return None
    return tuple(names[slice_obj])


def resolve_action_dim(info, vector_layout):
    action_dim = slice_length(vector_layout.action_slice)
    if action_dim is not None and action_dim > 0:
        return action_dim
    return get_feature_shape_dim(info, "action")


def collect_eef_target_candidates(robot_urdf):
    root = ET.parse(robot_urdf).getroot()
    matches = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name and (
            link_name.endswith("ee_gripper_link")
            or link_name.endswith("ee_arm_link")
        ):
            matches.append(link_name)

    candidates = []
    for link_name in sorted(set(matches)):
        try:
            required_dim = UrdfKinematics(robot_urdf, target_link=link_name).num_actuated_joints
        except ValueError:
            continue
        candidates.append((link_name, required_dim))

    return sorted(
        candidates,
        key=lambda item: (
            0 if item[0].endswith("ee_gripper_link") else 1,
            item[1],
            item[0],
        ),
    )


def parse_int_list(raw_value, arg_name):
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {arg_name}: {item!r}") from exc
    if not values:
        raise ValueError(f"{arg_name} must contain at least one integer.")
    return tuple(values)


def parse_optional_int_list(raw_value, arg_name):
    text = str(raw_value).strip().lower()
    if text in {"", "none", "null", "no", "false"}:
        return tuple()
    return parse_int_list(raw_value, arg_name)


def parse_string_list(raw_value):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one non-empty value.")
    return tuple(values)


def build_arm_slices(arm_joint_dims):
    arm_slices = []
    start = 0
    for arm_dim in arm_joint_dims:
        stop = start + int(arm_dim)
        arm_slices.append(slice(start, stop))
        start = stop
    return tuple(arm_slices)


def infer_default_gripper_indices(arm_slices):
    return tuple(arm_slice.stop - 1 for arm_slice in arm_slices if arm_slice.stop > arm_slice.start)


def infer_default_arm_fk_dims(arm_slices, gripper_indices):
    fk_dims = []
    for arm_slice in arm_slices:
        gripper_count = sum(1 for index in gripper_indices if arm_slice.start <= index < arm_slice.stop)
        fk_dims.append(max(0, (arm_slice.stop - arm_slice.start) - gripper_count))
    return tuple(fk_dims)


def infer_default_eef_target_link(robot_urdf, fk_dim):
    if fk_dim <= 0:
        return None
    compatible = collect_eef_target_candidates(robot_urdf)
    if not compatible:
        raise ValueError(
            "Could not infer EEF target link from URDF. Pass --eef_target_links explicitly."
        )

    for link_name, required_dim in compatible:
        if required_dim <= fk_dim:
            return link_name

    details = ", ".join(f"{name} requires {required_dim}" for name, required_dim in compatible)
    raise ValueError(
        "Could not infer an EEF target link compatible with the configured FK dimensions. "
        f"Pass --eef_target_links explicitly. Candidates: {details}"
    )


def expand_per_arm(values, expected_len, arg_name):
    if len(values) == expected_len:
        return tuple(values)
    if len(values) == 1:
        return tuple(values * expected_len)
    raise ValueError(f"{arg_name} must contain 1 or {expected_len} values, got {len(values)}.")


def camel_case_robot_name(robot_name):
    parts = re.findall(r"[A-Za-z0-9]+", str(robot_name))
    if not parts:
        return DEFAULT_ROBOT_NAME
    return "".join(part[:1].upper() + part[1:] for part in parts)


def extract_action_names(info, vector_layout, action_dim):
    action_names = get_feature_names(info, "action")
    if action_names is not None:
        action_names = slice_feature_names(action_names, vector_layout.action_slice) or action_names
        if len(action_names) == action_dim:
            return action_names

    state_names = get_feature_names(info, "observation.state")
    if state_names is not None and vector_layout.qpos_slice is not None:
        qpos_names = slice_feature_names(state_names, vector_layout.qpos_slice)
        if qpos_names is not None and len(qpos_names) == action_dim:
            return qpos_names

    return None


def infer_joint_group_label(name):
    text = str(name).strip().lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    if "left" in tokens or text.startswith("left_") or text.startswith("left/") or text.startswith("l_"):
        return "left"
    if "right" in tokens or text.startswith("right_") or text.startswith("right/") or text.startswith("r_"):
        return "right"

    for prefix in ("arm", "manipulator", "robot"):
        match = re.search(rf"(?:^|[^a-z0-9]){prefix}([0-9]+)(?:[^a-z0-9]|$)", text)
        if match:
            return f"{prefix}{match.group(1)}"
        match = re.match(rf"{prefix}([0-9]+)", text)
        if match:
            return f"{prefix}{match.group(1)}"

    return None


def infer_arm_joint_dims_from_names(action_names):
    if not action_names:
        return None

    labels = [infer_joint_group_label(name) for name in action_names]
    if any(label is None for label in labels):
        return None

    arm_joint_dims = []
    ordered_labels = []
    current_label = labels[0]
    current_count = 0
    for label in labels:
        if label != current_label:
            ordered_labels.append(current_label)
            arm_joint_dims.append(current_count)
            current_label = label
            current_count = 0
        current_count += 1
    ordered_labels.append(current_label)
    arm_joint_dims.append(current_count)

    if len(set(ordered_labels)) != len(ordered_labels):
        return None

    return tuple(arm_joint_dims)


def infer_gripper_indices_from_names(action_names):
    return tuple(
        index for index, name in enumerate(action_names)
        if GRIPPER_NAME_PATTERN.search(str(name))
    )


def infer_layout_from_robot_type(info, action_dim):
    robot_type_key = normalize_robot_type(info.get("robot_type")) if isinstance(info, dict) else ""
    if not robot_type_key:
        return {}

    for known_key, layout in KNOWN_ROBOT_LAYOUTS.items():
        if known_key in robot_type_key and sum(layout["arm_joint_dims"]) == action_dim:
            return dict(layout)
    return {}


def infer_layout_from_action_names(action_names):
    if not action_names:
        return {}

    arm_joint_dims = infer_arm_joint_dims_from_names(action_names)
    gripper_indices = infer_gripper_indices_from_names(action_names)
    if arm_joint_dims is None:
        return {}

    arm_slices = build_arm_slices(arm_joint_dims)
    if any(
        not any(candidate_slice.start <= index < candidate_slice.stop for candidate_slice in arm_slices)
        for index in gripper_indices
    ):
        return {}

    return {
        "arm_joint_dims": arm_joint_dims,
        "gripper_indices": gripper_indices,
        "arm_fk_dims": infer_default_arm_fk_dims(arm_slices, gripper_indices),
    }


def infer_layout_from_urdf(robot_urdf, action_dim):
    candidates = collect_eef_target_candidates(robot_urdf)
    solutions = []
    for target_link, required_dim in candidates:
        for has_gripper in (True, False):
            arm_dim = required_dim + (1 if has_gripper else 0)
            if arm_dim <= 0 or action_dim % arm_dim != 0:
                continue

            arm_count = action_dim // arm_dim
            if arm_count <= 0 or arm_count > 4:
                continue

            arm_joint_dims = tuple([arm_dim] * arm_count)
            arm_slices = build_arm_slices(arm_joint_dims)
            gripper_indices = infer_default_gripper_indices(arm_slices) if has_gripper else tuple()
            solutions.append(
                {
                    "target_link": target_link,
                    "arm_joint_dims": arm_joint_dims,
                    "gripper_indices": gripper_indices,
                    "arm_fk_dims": tuple([required_dim] * arm_count),
                    "eef_target_links": tuple([target_link] * arm_count),
                    "score": (
                        0 if has_gripper else 1,
                        0 if arm_count == 2 else 1,
                        abs(2 - arm_count),
                        arm_dim,
                    ),
                }
            )

    if not solutions:
        return {}

    best = min(solutions, key=lambda item: item["score"])
    return {
        "arm_joint_dims": best["arm_joint_dims"],
        "gripper_indices": best["gripper_indices"],
        "arm_fk_dims": best["arm_fk_dims"],
        "eef_target_links": best["eef_target_links"],
    }


def infer_robot_layout_defaults(info, robot_urdf, vector_layout):
    action_dim = resolve_action_dim(info, vector_layout)
    if action_dim is None or action_dim <= 0:
        return {}

    action_names = extract_action_names(info, vector_layout, action_dim)

    inferred = infer_layout_from_robot_type(info, action_dim)
    if not inferred:
        inferred = infer_layout_from_action_names(action_names)
    if not inferred:
        inferred = infer_layout_from_urdf(robot_urdf, action_dim)

    if "eef_target_links" not in inferred and "arm_fk_dims" in inferred:
        inferred["eef_target_links"] = tuple(
            infer_default_eef_target_link(robot_urdf, fk_dim)
            for fk_dim in inferred["arm_fk_dims"]
        )

    if action_names is not None:
        inferred.setdefault("action_names", action_names)

    return inferred


def resolve_robot_layout(info, robot_urdf, vector_layout, arm_joint_dims=None, gripper_indices=None, arm_fk_dims=None, eef_target_links=None, robot_name=None):
    inferred = infer_robot_layout_defaults(info, robot_urdf, vector_layout)

    if arm_joint_dims is None:
        arm_joint_dims = inferred.get("arm_joint_dims", DEFAULT_ARM_JOINT_DIMS)
    else:
        arm_joint_dims = parse_int_list(arm_joint_dims, "--arm_joint_dims")

    arm_slices = build_arm_slices(arm_joint_dims)

    if gripper_indices is None:
        gripper_indices = inferred.get("gripper_indices", infer_default_gripper_indices(arm_slices))
    else:
        gripper_indices = tuple(sorted(parse_optional_int_list(gripper_indices, "--gripper_indices")))

    if len(set(gripper_indices)) != len(gripper_indices):
        raise ValueError("--gripper_indices must not contain duplicates.")

    for index in gripper_indices:
        if index < 0 or index >= sum(arm_joint_dims):
            raise ValueError(
                f"gripper index {index} is out of range for total joint dim {sum(arm_joint_dims)}"
            )

    if arm_fk_dims is None:
        arm_fk_dims = inferred.get("arm_fk_dims", infer_default_arm_fk_dims(arm_slices, gripper_indices))
    else:
        arm_fk_dims = expand_per_arm(parse_int_list(arm_fk_dims, "--arm_fk_dims"), len(arm_slices), "--arm_fk_dims")

    for arm_index, (arm_slice, fk_dim) in enumerate(zip(arm_slices, arm_fk_dims)):
        arm_dim = arm_slice.stop - arm_slice.start
        if fk_dim < 0 or fk_dim > arm_dim:
            raise ValueError(
                f"arm_fk_dims[{arm_index}]={fk_dim} is out of range for arm joint dim {arm_dim}"
            )

    if eef_target_links is None:
        eef_target_links = inferred.get("eef_target_links")
        if eef_target_links is None:
            eef_target_links = tuple(
                infer_default_eef_target_link(robot_urdf, fk_dim)
                for fk_dim in arm_fk_dims
            )
    else:
        eef_target_links = expand_per_arm(parse_string_list(eef_target_links), len(arm_slices), "--eef_target_links")

    for arm_index, (target_link, fk_dim) in enumerate(zip(eef_target_links, arm_fk_dims)):
        if fk_dim <= 0 or target_link is None:
            continue
        required_dim = UrdfKinematics(robot_urdf, target_link=target_link).num_actuated_joints
        if required_dim > fk_dim:
            raise ValueError(
                f"EEF target link '{target_link}' for arm {arm_index} requires {required_dim} joints, "
                f"but arm_fk_dims[{arm_index}] is {fk_dim}."
            )

    detected_robot_name = None
    if isinstance(info, dict):
        detected_robot_name = info.get("robot_type")
    robot_name = camel_case_robot_name(robot_name or detected_robot_name or DEFAULT_ROBOT_NAME)

    return RobotLayout(
        arm_joint_dims=tuple(arm_joint_dims),
        gripper_indices=tuple(gripper_indices),
        arm_fk_dims=tuple(arm_fk_dims),
        arm_slices=arm_slices,
        eef_target_links=tuple(eef_target_links),
        robot_name=robot_name,
    )


def format_slice(slice_obj):
    if slice_obj is None:
        return "all"
    return f"{slice_obj.start}:{slice_obj.stop}"


def list_parquet_columns(parquet_path):
    schema = pq.ParquetFile(parquet_path).schema_arrow
    return tuple((field.name, str(field.type)) for field in schema)


def build_mapping_lines(info, vector_layout, robot_layout):
    action_dim = resolve_action_dim(info, vector_layout)
    action_names = extract_action_names(info, vector_layout, action_dim) if action_dim is not None else None
    qpos_dim = slice_length(vector_layout.qpos_slice) or action_dim
    video_specs = list_video_specs(info)

    lines = []
    lines.append(
        f"observation.state[{format_slice(vector_layout.qpos_slice)}]"
        f" -> measured_joint_pos"
        + (f" ({qpos_dim} dims)" if qpos_dim is not None else "")
    )

    if vector_layout.qvel_slice is not None:
        lines.append(f"observation.state[{format_slice(vector_layout.qvel_slice)}] -> measured_joint_vel")
    else:
        lines.append("observation.state[qvel] -> measured_joint_vel (zero-filled when absent)")

    if vector_layout.effort_slice is not None:
        lines.append(
            f"observation.state[{format_slice(vector_layout.effort_slice)}] -> measured_eef_wrench "
            "(per-arm first min(6, fk_dim) dims)"
        )
    else:
        lines.append("observation.state[effort] -> measured_eef_wrench (zero-filled when absent)")

    lines.append(
        f"action[{format_slice(vector_layout.action_slice)}] -> command_joint_pos"
        + (f" ({action_dim} dims)" if action_dim is not None else "")
    )

    if robot_layout.gripper_indices:
        lines.append(
            f"joint dims {robot_layout.gripper_indices} -> "
            "measured_gripper_joint_pos / command_gripper_joint_pos"
        )
    else:
        lines.append("gripper joints -> none")

    fk_pairs = list(zip(robot_layout.arm_slices, robot_layout.arm_fk_dims, robot_layout.eef_target_links))
    fk_parts = []
    for arm_index, (arm_slice, fk_dim, target_link) in enumerate(fk_pairs):
        if fk_dim <= 0 or target_link is None:
            fk_parts.append(f"arm{arm_index}[{arm_slice.start}:{arm_slice.stop}] => zero EEF")
        else:
            fk_parts.append(
                f"arm{arm_index}[{arm_slice.start}:{arm_slice.stop}] "
                f"fk={fk_dim} -> {target_link}"
            )
    lines.append(
        "; ".join(fk_parts)
        + " -> measured_eef_pose / command_eef_pose"
    )

    if "timestamp" in info_features(info):
        lines.append("timestamp -> time")
    else:
        lines.append("timestamp -> time (derived from fps/sample index when absent)")

    lines.append("task_index + meta/tasks.jsonl + meta/episodes.jsonl -> task_desc")

    if video_specs:
        for spec in video_specs:
            suffix = "depth_image" if spec["is_depth"] else "rgb_image"
            lines.append(f"{spec['feature_key']} -> {spec['camera_name']}_{suffix}.rmb.mp4")
    else:
        lines.append("observation.images.* -> no video export")

    if action_names:
        lines.append("action joint names: " + ", ".join(action_names))

    return lines


def print_dataset_mapping_summary(dataset_name, info, vector_layout, robot_layout, parquet_columns=None):
    print(f"\n🧾 LeRobot keys: {dataset_name}")
    for feature_key, feature_spec in info_features(info).items():
        dtype = feature_spec.get("dtype", "?") if isinstance(feature_spec, dict) else "?"
        shape = feature_spec.get("shape", "?") if isinstance(feature_spec, dict) else "?"
        print(f"  - {feature_key} (dtype={dtype}, shape={shape})")

    if parquet_columns is not None:
        print("🧱 Parquet schema:")
        for column_name, column_type in parquet_columns:
            print(f"  - {column_name} ({column_type})")

    print("🔁 LeRobot -> RMB mapping:")
    for line in build_mapping_lines(info, vector_layout, robot_layout):
        print(f"  - {line}")


def load_dataset_bundle(
    dataset_dir,
    robot_urdf,
    arm_joint_dims=None,
    gripper_indices=None,
    arm_fk_dims=None,
    eef_target_links=None,
    robot_name=None,
):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_name = dataset_dir.name
    info, episodes_meta, tasks_by_index, modality = load_dataset_metadata(dataset_dir)
    vector_layout = resolve_vector_layout(info, modality)
    robot_layout = resolve_robot_layout(
        info=info,
        robot_urdf=robot_urdf,
        vector_layout=vector_layout,
        arm_joint_dims=arm_joint_dims,
        gripper_indices=gripper_indices,
        arm_fk_dims=arm_fk_dims,
        eef_target_links=eef_target_links,
        robot_name=robot_name,
    )
    parquet_files = tuple(list_episode_parquet_files(dataset_dir))
    video_specs = tuple(list_video_specs(info))
    parquet_columns = list_parquet_columns(parquet_files[0]) if parquet_files else tuple()

    return DatasetBundle(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        info=info,
        episodes_meta=episodes_meta,
        tasks_by_index=tasks_by_index,
        vector_layout=vector_layout,
        robot_layout=robot_layout,
        parquet_files=parquet_files,
        parquet_columns=parquet_columns,
        video_specs=video_specs,
    )


def print_dataset_overview(bundle):
    print(f"\n📦 Processing dataset: {bundle.dataset_name}")
    print(
        "🤖 Robot layout: "
        f"arms={bundle.robot_layout.arm_joint_dims}, "
        f"grippers={bundle.robot_layout.gripper_indices}, "
        f"fk={bundle.robot_layout.arm_fk_dims}, "
        f"eef={bundle.robot_layout.eef_target_links}"
    )
    if any(
        value is not None
        for value in (
            bundle.vector_layout.qpos_slice,
            bundle.vector_layout.qvel_slice,
            bundle.vector_layout.effort_slice,
            bundle.vector_layout.action_slice,
        )
    ):
        print(
            "🧭 Vector layout: "
            f"qpos={bundle.vector_layout.qpos_slice}, "
            f"qvel={bundle.vector_layout.qvel_slice}, "
            f"effort={bundle.vector_layout.effort_slice}, "
            f"action={bundle.vector_layout.action_slice}"
        )
    print_dataset_mapping_summary(
        bundle.dataset_name,
        bundle.info,
        bundle.vector_layout,
        bundle.robot_layout,
        parquet_columns=bundle.parquet_columns,
    )


def build_episode_jobs(bundle, out_dir, requested_fps, camera_workers, video_preset, robot_urdf):
    out_dir = Path(out_dir).expanduser().resolve()
    return [
        EpisodeJob(
            dataset_dir=bundle.dataset_dir,
            dataset_name=bundle.dataset_name,
            parquet_path=parquet_path,
            out_dir=out_dir,
            info=bundle.info,
            episodes_meta=bundle.episodes_meta,
            tasks_by_index=bundle.tasks_by_index,
            vector_layout=bundle.vector_layout,
            requested_fps=requested_fps,
            camera_workers=camera_workers,
            video_preset=video_preset,
            robot_urdf=str(robot_urdf),
            robot_layout=bundle.robot_layout,
            video_specs=bundle.video_specs,
        )
        for parquet_path in bundle.parquet_files
    ]


def confirm_dataset_conversion(dataset_name, assume_yes=False):
    if assume_yes:
        return True

    if not sys.stdin.isatty():
        print(
            f"❌ Confirmation required before converting dataset '{dataset_name}', "
            "but stdin is not interactive. Re-run with --yes to proceed."
        )
        return False

    while True:
        answer = input(f"Convert dataset '{dataset_name}'? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def compute_eef_pose_sequence(joint_sequence, kinematics, fk_dim):
    joint_sequence = np.asarray(joint_sequence, dtype=np.float64)
    poses = np.zeros((joint_sequence.shape[0], 7), dtype=np.float64)
    if fk_dim <= 0 or kinematics is None:
        poses[:, 3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return poses
    for index, joints in enumerate(joint_sequence):
        transform = kinematics.forward_kinematics(joints[:fk_dim])
        poses[index, :3] = transform[:3, 3]
        poses[index, 3:] = quaternion_xyzw_from_rotation(transform[:3, :3])
    return poses


def build_multiarm_eef_data(joint_sequence, kinematics_per_arm, layout):
    poses_per_arm = []
    rel_per_arm = []
    for arm_slice, fk_dim, kinematics in zip(layout.arm_slices, layout.arm_fk_dims, kinematics_per_arm):
        arm_joints = joint_sequence[:, arm_slice]
        arm_poses = compute_eef_pose_sequence(arm_joints, kinematics, fk_dim)
        poses_per_arm.append(arm_poses)
        rel_per_arm.append(compute_eef_pose_rel(arm_poses))
    return np.concatenate(poses_per_arm, axis=1), np.concatenate(rel_per_arm, axis=1)


def build_multiarm_eef_wrench_data(effort_sequence, layout):
    effort_sequence = np.asarray(effort_sequence, dtype=np.float64)
    wrench_per_arm = []
    for arm_slice, fk_dim in zip(layout.arm_slices, layout.arm_fk_dims):
        arm_effort = effort_sequence[:, arm_slice]
        wrench = np.zeros((len(effort_sequence), 6), dtype=np.float64)
        used_dim = min(6, fk_dim, arm_effort.shape[1])
        if used_dim > 0:
            wrench[:, :used_dim] = arm_effort[:, :used_dim]
        wrench_per_arm.append(wrench)
    return np.concatenate(wrench_per_arm, axis=1)


def save_episode_hdf5_compatible(
    out_path,
    task_desc,
    qpos,
    qvel,
    action,
    time_values,
    camera_names,
    kinematics_per_arm,
    layout,
    effort=None,
):
    dtype_f8 = np.float64
    num_steps = len(time_values)
    num_grippers = len(layout.gripper_indices)
    num_eef = len(layout.arm_slices)

    if qpos.shape[1] < layout.total_arm_joint_dim or action.shape[1] < layout.total_arm_joint_dim:
        raise ValueError(
            f"Expected at least {layout.total_arm_joint_dim} joint dimensions for configured robot layout, "
            f"got qpos={qpos.shape[1]}, action={action.shape[1]}"
        )

    measured_joint_pos = qpos.astype(dtype_f8)
    measured_joint_vel = qvel.astype(dtype_f8)
    command_joint_pos = action.astype(dtype_f8)

    if num_grippers > 0:
        measured_gripper_joint_pos = qpos[:, layout.gripper_indices].astype(dtype_f8)
        command_gripper_joint_pos = action[:, layout.gripper_indices].astype(dtype_f8)
    else:
        measured_gripper_joint_pos = np.zeros((num_steps, 0), dtype=dtype_f8)
        command_gripper_joint_pos = np.zeros((num_steps, 0), dtype=dtype_f8)

    measured_joint_pos_rel = previous_step_delta(measured_joint_pos)
    command_joint_pos_rel = previous_step_delta(command_joint_pos)
    measured_gripper_joint_pos_rel = previous_step_delta(measured_gripper_joint_pos)
    command_gripper_joint_pos_rel = previous_step_delta(command_gripper_joint_pos)
    measured_eef_pose, measured_eef_pose_rel = build_multiarm_eef_data(measured_joint_pos, kinematics_per_arm, layout)
    command_eef_pose, command_eef_pose_rel = build_multiarm_eef_data(command_joint_pos, kinematics_per_arm, layout)
    if effort is not None:
        effort = np.asarray(effort, dtype=dtype_f8)
        if effort.shape[1] < layout.total_arm_joint_dim:
            raise ValueError(
                f"Expected at least {layout.total_arm_joint_dim} effort dimensions for configured robot layout, got {effort.shape[1]}"
            )
        measured_eef_wrench = build_multiarm_eef_wrench_data(effort, layout)
    else:
        measured_eef_wrench = np.zeros((num_steps, 6 * num_eef), dtype=dtype_f8)
    zeros_reward = np.zeros((num_steps,), dtype=dtype_f8)

    with h5py.File(out_path, "w") as f:
        f.attrs["camera_names"] = string_array(camera_names)
        f.attrs["demo_name"] = f"{layout.robot_name}RmbDemo"
        f.attrs["env"] = f"{layout.robot_name}RmbDemoEnv"
        f.attrs["format"] = "RmbData-Compact"
        f.attrs["task_desc"] = task_desc
        f.attrs["version"] = "3.0.0"
        f.attrs["world_idx"] = 0
        f.attrs["pointcloud_camera_names"] = string_array([])
        f.attrs["rgb_tactile_names"] = string_array([])

        f.create_dataset("command_eef_pose", data=command_eef_pose)
        f.create_dataset("command_eef_pose_rel", data=command_eef_pose_rel)
        f.create_dataset("command_gripper_joint_pos", data=command_gripper_joint_pos)
        f.create_dataset("command_gripper_joint_pos_rel", data=command_gripper_joint_pos_rel)
        f.create_dataset("command_joint_pos", data=command_joint_pos)
        f.create_dataset("command_joint_pos_rel", data=command_joint_pos_rel)
        f.create_dataset("measured_eef_pose", data=measured_eef_pose)
        f.create_dataset("measured_eef_pose_rel", data=measured_eef_pose_rel)
        f.create_dataset("measured_eef_wrench", data=measured_eef_wrench)
        f.create_dataset("measured_gripper_joint_pos", data=measured_gripper_joint_pos)
        f.create_dataset("measured_gripper_joint_pos_rel", data=measured_gripper_joint_pos_rel)
        f.create_dataset("measured_joint_pos", data=measured_joint_pos)
        f.create_dataset("measured_joint_pos_rel", data=measured_joint_pos_rel)
        f.create_dataset("measured_joint_vel", data=measured_joint_vel)
        f.create_dataset("reward", data=zeros_reward)
        f.create_dataset("time", data=time_values.astype(dtype_f8))


def load_episode_from_parquet(parquet_path):
    schema = pq.ParquetFile(parquet_path).schema_arrow.names
    observation_column = resolve_column_name(schema, exact="observation.state", prefix="observation.state.")
    action_column = resolve_column_name(schema, exact="action", prefix="action.")

    optional_columns = [name for name in ("timestamp", "task_index") if name in schema]
    table = pq.read_table(parquet_path, columns=[observation_column, action_column] + optional_columns)

    observation_state = stack_list_column(table[observation_column], dtype=np.float64)
    action = stack_list_column(table[action_column], dtype=np.float64)

    timestamps = None
    if "timestamp" in table.column_names:
        timestamps = scalar_column_to_numpy(table["timestamp"], dtype=np.float64)

    task_indices = None
    if "task_index" in table.column_names:
        task_indices = scalar_column_to_numpy(table["task_index"], dtype=np.int64)

    return observation_state, action, timestamps, task_indices


def apply_action_slice(action, vector_layout, parquet_path):
    action = np.asarray(action, dtype=np.float64)
    action_slice = vector_layout.action_slice
    if action_slice is None:
        return action

    if action_slice.stop is None or action.shape[1] < action_slice.stop:
        raise ValueError(
            f"action slice {action_slice.start}:{action_slice.stop} from modality.json is out of range for "
            f"{parquet_path.name}: action_dim={action.shape[1]}"
        )
    return action[:, action_slice]


def slice_or_none(arr, slice_obj):
    if slice_obj is None:
        return None
    if slice_obj.stop is None or arr.shape[1] < slice_obj.stop:
        return None
    return arr[:, slice_obj]


def split_state_components(observation_state, action_dim, parquet_path, vector_layout):
    obs_dim = observation_state.shape[1]
    if obs_dim < action_dim:
        raise ValueError(
            f"observation.state is smaller than action for {parquet_path.name}: "
            f"obs_dim={obs_dim}, action_dim={action_dim}"
        )

    qpos = slice_or_none(observation_state, vector_layout.qpos_slice)
    qvel = slice_or_none(observation_state, vector_layout.qvel_slice)
    effort = slice_or_none(observation_state, vector_layout.effort_slice)

    if qpos is not None:
        if qpos.shape[1] != action_dim:
            print(
                f"⚠️  qpos dim from modality.json for {parquet_path.name} is {qpos.shape[1]}, "
                f"while action dim is {action_dim}. Continuing with qpos dimensions from modality.json."
            )
        if qvel is None:
            qvel = np.zeros_like(qpos)
            print(f"⚠️  modality.json for {parquet_path.name} does not define qvel; zero-filling measured_joint_vel.")
        if effort is None:
            print(f"⚠️  modality.json for {parquet_path.name} does not define effort; measured_eef_wrench will be zero-filled.")
        return qpos, qvel, effort

    qpos = observation_state[:, :action_dim]

    if obs_dim >= 3 * action_dim:
        qvel = observation_state[:, action_dim : 2 * action_dim]
        effort = observation_state[:, 2 * action_dim : 3 * action_dim]
    elif obs_dim >= 2 * action_dim:
        qvel = observation_state[:, action_dim : 2 * action_dim]
        effort = None
        print(
            f"⚠️  observation.state in {parquet_path.name} does not include effort; "
            "measured_eef_wrench will be zero-filled."
        )
    else:
        qvel = np.zeros_like(qpos)
        effort = None
        print(
            f"⚠️  observation.state in {parquet_path.name} only contains qpos-like values; "
            "qvel and measured_eef_wrench will be zero-filled."
        )

    if obs_dim > 3 * action_dim:
        print(
            f"ℹ️  observation.state in {parquet_path.name} has trailing dimensions "
            f"({obs_dim} total). Only the first {3 * action_dim} dims are used as qpos/qvel/effort."
        )

    return qpos, qvel, effort


def resolve_video_path(dataset_dir, info, parquet_path, episode_index, feature_key):
    chunk_index = infer_chunk_index(parquet_path)
    video_path_template = info.get("video_path")
    if isinstance(video_path_template, str) and video_path_template:
        candidate = dataset_dir / video_path_template.format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
            video_key=feature_key,
        )
        if candidate.exists():
            return candidate

    fallback = dataset_dir / "videos" / f"chunk-{chunk_index:03d}" / feature_key / f"episode_{episode_index:06d}.mp4"
    if fallback.exists():
        return fallback

    return None


def load_video_frames(video_path, is_depth, camera_name):
    if is_depth:
        try:
            return videoio.uint16read(video_path)
        except Exception:
            rgb_frames = videoio.videoread(video_path)
            print(
                f"⚠️  Depth video {video_path.name} could not be decoded as uint16. "
                f"Falling back to 8-bit conversion for camera '{camera_name}'."
            )
            if rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3:
                return convert_depth_frames(rgb_frames[..., 0], camera_name)
            return convert_depth_frames(rgb_frames, camera_name)

    return videoio.videoread(video_path)


def sample_video_frames(frames, sample_indices):
    if len(frames) == 0:
        raise ValueError("Video contains no frames.")
    clipped = np.clip(sample_indices, 0, len(frames) - 1)
    return frames[clipped]


def export_camera_stream(dataset_dir, info, parquet_path, episode_index, sample_indices, fps, rmb_dir, spec, video_preset):
    video_path = resolve_video_path(dataset_dir, info, parquet_path, episode_index, spec["feature_key"])
    if video_path is None:
        print(
            f"⚠️  Video for '{spec['feature_key']}' was not found for episode_{episode_index:06d}; "
            "skipping camera export."
        )
        return None

    frames = load_video_frames(video_path, spec["is_depth"], spec["camera_name"])
    sampled_frames = sample_video_frames(frames, sample_indices)

    if spec["is_depth"]:
        video_path_out = rmb_dir / f"{spec['camera_name']}_depth_image.rmb.mp4"
        print(f"🎞️ Saving video: {video_path_out.name}")
        videoio.uint16save(video_path_out, sampled_frames, preset=video_preset, fps=fps)
    else:
        video_path_out = rmb_dir / f"{spec['camera_name']}_rgb_image.rmb.mp4"
        print(f"🎞️ Saving video: {video_path_out.name}")
        videoio.videosave(video_path_out, sampled_frames, lossless=False, preset=video_preset, fps=fps)

    return spec["camera_name"]


def export_episode_videos(
    dataset_dir,
    info,
    parquet_path,
    episode_index,
    sample_indices,
    fps,
    rmb_dir,
    video_specs,
    camera_workers,
    video_preset,
):
    if not video_specs:
        return []

    exported = []
    max_workers = camera_workers
    if max_workers <= 0:
        max_workers = min(len(video_specs), os.cpu_count() or 1)

    if max_workers > 1 and len(video_specs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    export_camera_stream,
                    dataset_dir,
                    info,
                    parquet_path,
                    episode_index,
                    sample_indices,
                    fps,
                    rmb_dir,
                    spec,
                    video_preset,
                )
                for spec in video_specs
            ]
            for future in futures:
                camera_name = future.result()
                if camera_name is not None:
                    exported.append(camera_name)
    else:
        for spec in video_specs:
            camera_name = export_camera_stream(
                dataset_dir,
                info,
                parquet_path,
                episode_index,
                sample_indices,
                fps,
                rmb_dir,
                spec,
                video_preset,
            )
            if camera_name is not None:
                exported.append(camera_name)

    return exported


def episode_output_name(episode_index):
    return f"episode_{episode_index:06d}.rmb"


def process_single_episode(job):
    episode_index = infer_episode_index(job.parquet_path)
    episode_name = episode_output_name(episode_index)
    rmb_dir = job.out_dir / job.dataset_name / episode_name
    rmb_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Reading parquet: {job.parquet_path}")

    observation_state, action, timestamps, task_indices = load_episode_from_parquet(job.parquet_path)
    action = apply_action_slice(action, job.vector_layout, job.parquet_path)
    if action.shape[1] != job.robot_layout.total_arm_joint_dim:
        print(
            f"⚠️  action dim for {job.parquet_path.name} is {action.shape[1]}. "
            f"This converter uses the first {job.robot_layout.total_arm_joint_dim} dims based on the configured robot layout."
        )
    qpos, qvel, effort = split_state_components(observation_state, action.shape[1], job.parquet_path, job.vector_layout)

    source_fps = resolve_source_fps(job.info, timestamps)
    target_fps = resolve_target_fps(source_fps, job.requested_fps)
    sample_indices = build_sample_indices(len(action), source_fps, target_fps)

    qpos_rs = qpos[sample_indices]
    qvel_rs = qvel[sample_indices]
    effort_rs = effort[sample_indices] if effort is not None else None
    action_rs = action[sample_indices]

    if timestamps is not None and len(timestamps) == len(action):
        time_values = timestamps[sample_indices]
    else:
        time_values = sample_indices.astype(np.float64) / source_fps

    task_desc = resolve_episode_task_desc(
        dataset_name=job.dataset_name,
        episode_index=episode_index,
        episodes_meta=job.episodes_meta,
        tasks_by_index=job.tasks_by_index,
        task_indices=task_indices,
    )

    camera_names = export_episode_videos(
        dataset_dir=job.dataset_dir,
        info=job.info,
        parquet_path=job.parquet_path,
        episode_index=episode_index,
        sample_indices=sample_indices,
        fps=target_fps,
        rmb_dir=rmb_dir,
        video_specs=job.video_specs,
        camera_workers=job.camera_workers,
        video_preset=job.video_preset,
    )

    save_episode_hdf5_compatible(
        rmb_dir / "main.rmb.hdf5",
        task_desc=task_desc,
        qpos=qpos_rs,
        qvel=qvel_rs,
        action=action_rs,
        effort=effort_rs,
        time_values=time_values,
        camera_names=camera_names,
        kinematics_per_arm=[
            (
                UrdfKinematics(job.robot_urdf, target_link=target_link)
                if target_link is not None else None
            )
            for target_link in job.robot_layout.eef_target_links
        ],
        layout=job.robot_layout,
    )

    print(f"✅ Done: {episode_name}")


def process_dataset(
    input_dir,
    out_dir,
    fps=None,
    nproc=1,
    camera_workers=0,
    video_preset="veryfast",
    robot_urdf=None,
    arm_joint_dims=None,
    gripper_indices=None,
    arm_fk_dims=None,
    eef_target_links=None,
    robot_name=None,
    assume_yes=False,
):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_urdf = resolve_robot_urdf(robot_urdf)

    dataset_dirs = discover_lerobot_datasets(input_dir)
    if not dataset_dirs:
        print(f"❌ No LeRobot datasets found under: {input_dir}")
        return

    for dataset_dir in dataset_dirs:
        bundle = load_dataset_bundle(
            dataset_dir=dataset_dir,
            robot_urdf=robot_urdf,
            arm_joint_dims=arm_joint_dims,
            gripper_indices=gripper_indices,
            arm_fk_dims=arm_fk_dims,
            eef_target_links=eef_target_links,
            robot_name=robot_name,
        )

        if not bundle.parquet_files:
            print(f"⚠️  No parquet episodes found in dataset: {bundle.dataset_dir}")
            continue

        print_dataset_overview(bundle)
        if not confirm_dataset_conversion(bundle.dataset_name, assume_yes=assume_yes):
            print(f"⏭️  Skipped dataset: {bundle.dataset_name}")
            continue
        jobs = build_episode_jobs(bundle, out_dir, fps, camera_workers, video_preset, robot_urdf)

        if nproc > 1:
            with Pool(nproc) as pool:
                pool.map(process_single_episode, jobs)
        else:
            for job in jobs:
                process_single_episode(job)


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot episodes into RMB-compatible output.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to a LeRobot dataset folder or a root folder that contains multiple LeRobot datasets.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video/data FPS. Defaults to the FPS stored in LeRobot meta/info.json.",
    )
    parser.add_argument("--nproc", type=int, default=1, help="Number of parallel episode workers.")
    parser.add_argument(
        "--camera_workers",
        type=int,
        default=0,
        help="Number of parallel workers for camera export within a single episode. 0 uses CPU-count-based auto.",
    )
    parser.add_argument(
        "--video_preset",
        type=str,
        default="veryfast",
        help="ffmpeg preset passed through videoio, e.g. ultrafast, veryfast, medium, slow.",
    )
    parser.add_argument(
        "--robot_urdf",
        type=str,
        default=None,
        help="Path to the single-arm robot URDF used to compute EEF poses. Defaults to config/assets vx300s URDF.",
    )
    parser.add_argument(
        "--arm_joint_dims",
        type=str,
        default=None,
        help="Comma-separated joint dims per arm in observation/action order. Default is '7,7' for ALOHA.",
    )
    parser.add_argument(
        "--gripper_indices",
        type=str,
        default=None,
        help="Comma-separated global gripper joint indices. Default is the last joint of each arm slice. Use 'none' for gripperless robots.",
    )
    parser.add_argument(
        "--arm_fk_dims",
        type=str,
        default=None,
        help="Comma-separated FK joint dims per arm. Default excludes configured gripper joints from each arm.",
    )
    parser.add_argument(
        "--eef_target_links",
        type=str,
        default=None,
        help="Comma-separated URDF target links for FK. A single value is reused for every arm.",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default=None,
        help="Robot name used for RMB metadata attrs such as demo_name/env. Defaults to LeRobot meta.robot_type or Aloha.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the y/n confirmation prompt after showing the LeRobot keys and RMB mapping.",
    )
    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        out_dir=args.output_dir,
        fps=args.fps,
        nproc=args.nproc,
        camera_workers=args.camera_workers,
        video_preset=args.video_preset,
        robot_urdf=args.robot_urdf,
        arm_joint_dims=args.arm_joint_dims,
        gripper_indices=args.gripper_indices,
        arm_fk_dims=args.arm_fk_dims,
        eef_target_links=args.eef_target_links,
        robot_name=args.robot_name,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    main()
