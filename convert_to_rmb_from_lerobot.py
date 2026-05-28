#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
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
    quaternion_to_rotation_matrix,
    quaternion_xyzw_from_rotation,
    resolve_robot_urdf,
    rotation_vector_from_matrix,
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
EEF_TARGETS = {
    "measured_eef_pose",
    "command_eef_pose",
    "measured_eef_pose_rel",
    "command_eef_pose_rel",
}
AXIS_NAME_TO_INDEX = {"x": 0, "y": 1, "z": 2}


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
class AxisTransform:
    perm: tuple[int, int, int]
    signs: tuple[float, float, float]
    text: str


@dataclass(frozen=True)
class SourceSelection:
    source: str
    indices: tuple[int, ...]
    field_names: tuple[str, ...] | None
    selector_text: str
    label: str | None = None
    transform: AxisTransform | None = None

    @property
    def dim(self):
        return len(self.indices)


@dataclass(frozen=True)
class TargetMapping:
    target: str
    groups: tuple[SourceSelection, ...]
    operation: str = "concat"

    @property
    def dim(self):
        if self.operation == "xyz_pair_distance":
            return len(self.groups) // 2
        return sum(group.dim for group in self.groups)


@dataclass(frozen=True)
class MappingConfig:
    path: str | None
    vector_layout_overrides: dict
    robot_layout_overrides: dict
    default_target_policies: dict[str, str]
    target_mappings: dict[str, TargetMapping]
    eef_transform: AxisTransform | None = None


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    dataset_name: str
    info: dict
    episodes_meta: dict
    tasks_by_index: dict
    vector_layout: VectorLayout
    robot_layout: RobotLayout
    mapping_config: MappingConfig
    parquet_files: tuple[Path, ...]
    parquet_columns: tuple[tuple[str, str], ...]
    parquet_vector_dims: tuple[tuple[str, int], ...]
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
    mapping_config: MappingConfig
    video_specs: tuple[dict, ...]
    skip_static_eef: bool = False


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


def read_optional_json(path):
    if path is None:
        return {}
    resolved = Path(path).expanduser().resolve()
    return read_json(resolved)


def parse_eef_transform_axes_arg(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "identity"}:
        return None
    axes = [axis.strip() for axis in value.split(",")]
    if len(axes) != 3:
        raise ValueError("--eef_transform_axes must contain three comma-separated axes, e.g. x,-z,y")
    return {"axes": axes}


def looks_like_eef_transform_axes_token(value):
    parts = [part.strip().lower() for part in str(value).split(",")]
    if len(parts) != 3:
        return False
    for part in parts:
        if not part:
            return False
        axis_name = part[1:] if part[:1] in ("-", "+") else part
        if axis_name not in AXIS_NAME_TO_INDEX:
            return False
    return True


def normalize_cli_args(argv):
    normalized = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--eef_transform_axes" and index + 1 < len(argv):
            next_token = argv[index + 1]
            if next_token.startswith("-") and looks_like_eef_transform_axes_token(next_token):
                normalized.append(f"--eef_transform_axes={next_token}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


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


def parse_slice_spec(spec, context):
    if spec is None:
        return None

    if isinstance(spec, dict):
        if "start" not in spec or "end" not in spec:
            raise ValueError(f"{context} must contain both 'start' and 'end'.")
        start = spec["start"]
        end = spec["end"]
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        start, end = spec
    else:
        raise ValueError(f"{context} must be a {{start, end}} object or a [start, end] pair.")

    try:
        start = int(start)
        end = int(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must use integer bounds.") from exc

    if start < 0 or end < start:
        raise ValueError(f"{context} must satisfy 0 <= start <= end.")
    return slice(start, end)


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


def parse_bool_metadata(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
                "is_depth": parse_bool_metadata(video_info.get("video.is_depth_map", False)),
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


def resolve_optional_column_name(schema_names, exact, prefix):
    try:
        return resolve_column_name(schema_names, exact=exact, prefix=prefix)
    except KeyError:
        return None


def mapped_source_names(mapping_config):
    if mapping_config is None:
        return set()

    sources = set()
    for mapping in mapping_config.target_mappings.values():
        for group in mapping.groups:
            sources.add(group.source)
    return sources


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


def source_feature_names(info, source):
    if get_feature_spec(info, source) is None:
        raise ValueError(f"Unsupported mapping source: {source!r}")
    return get_feature_names(info, source)


def parse_vector_layout_overrides(config):
    if not isinstance(config, dict):
        return {}

    overrides = {}
    for key in ("qpos", "qvel", "effort", "action"):
        if key not in config:
            continue
        value = config[key]
        if value is None:
            overrides[key] = None
        else:
            overrides[key] = parse_slice_spec(value, f"vector_layout.{key}")
    return overrides


def parse_axis_transform(config, context):
    if config is None:
        return None
    if isinstance(config, AxisTransform):
        return config
    if not isinstance(config, dict):
        raise ValueError(f"{context}.transform must be an object.")

    axes = config.get("axes")
    if axes is None:
        perm_raw = config.get("perm")
        signs_raw = config.get("signs", [1, 1, 1])
        if perm_raw is None:
            return None
        if not isinstance(perm_raw, list) or len(perm_raw) != 3:
            raise ValueError(f"{context}.transform.perm must be a 3-element list.")
        if not isinstance(signs_raw, list) or len(signs_raw) != 3:
            raise ValueError(f"{context}.transform.signs must be a 3-element list.")

        perm = []
        for axis in perm_raw:
            axis_name = str(axis).strip().lower()
            if axis_name not in AXIS_NAME_TO_INDEX:
                raise ValueError(f"{context}.transform.perm contains unsupported axis {axis!r}.")
            perm.append(AXIS_NAME_TO_INDEX[axis_name])
        signs = [float(sign) for sign in signs_raw]
        text = " ".join(f"{sign:+g}{'xyz'[index]}" for index, sign in zip(perm, signs))
    else:
        if not isinstance(axes, list) or len(axes) != 3:
            raise ValueError(f"{context}.transform.axes must be a 3-element list, e.g. ['x', '-z', 'y'].")
        perm = []
        signs = []
        tokens = []
        for raw_axis in axes:
            token = str(raw_axis).strip().lower()
            sign = -1.0 if token.startswith("-") else 1.0
            axis_name = token[1:] if token[:1] in ("-", "+") else token
            if axis_name not in AXIS_NAME_TO_INDEX:
                raise ValueError(f"{context}.transform.axes contains unsupported axis {raw_axis!r}.")
            perm.append(AXIS_NAME_TO_INDEX[axis_name])
            signs.append(sign)
            tokens.append(("-" if sign < 0 else "") + axis_name)
        text = "[" + ", ".join(tokens) + "]"

    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"{context}.transform must reference x, y, z exactly once.")
    if any(not np.isclose(abs(sign), 1.0) for sign in signs):
        raise ValueError(f"{context}.transform signs must be +1 or -1.")

    return AxisTransform(perm=tuple(perm), signs=tuple(signs), text=text)


def resolve_selection_spec(spec, info, context, inherited_source=None, inherited_transform=None):
    if not isinstance(spec, dict):
        raise ValueError(f"{context} must be an object.")

    source = spec.get("source", inherited_source)
    if source is None:
        raise ValueError(f"{context} must define 'source'.")
    source = str(source)

    present_keys = [key for key in ("fields", "indices", "slice") if key in spec]
    if len(present_keys) != 1:
        raise ValueError(f"{context} must define exactly one of: fields, indices, slice.")

    selector_key = present_keys[0]
    field_names = None

    if selector_key == "fields":
        fields = spec["fields"]
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{context}.fields must be a non-empty list.")

        source_spec = get_feature_spec(info, source)
        if source_spec is None:
            indices = list(range(len(fields)))
            normalized_fields = [str(field) for field in fields]
            field_names = tuple(normalized_fields)
            selector_text = summarize_names(field_names) or ", ".join(field_names)
        else:
            source_names = get_feature_names(info, source)
            if source_names is None:
                raise ValueError(
                    f"{context} uses named fields, but {source!r} does not expose per-dimension names in meta/info.json."
                )

            name_to_indices = {}
            for index, name in enumerate(source_names):
                name_to_indices.setdefault(str(name), []).append(index)

            indices = []
            normalized_fields = []
            for field in fields:
                field_name = str(field)
                matches = name_to_indices.get(field_name)
                if not matches:
                    raise ValueError(f"{context} refers to unknown field {field_name!r} in {source!r}.")
                if len(matches) > 1:
                    raise ValueError(f"{context} refers to ambiguous field {field_name!r} in {source!r}.")
                indices.append(matches[0])
                normalized_fields.append(field_name)

            field_names = tuple(normalized_fields)
            selector_text = summarize_names(field_names) or ", ".join(field_names)
    elif selector_key == "indices":
        raw_indices = spec["indices"]
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError(f"{context}.indices must be a non-empty list.")
        indices = tuple(parse_int_list(raw_indices, f"{context}.indices"))
        selector_text = "[" + ", ".join(str(index) for index in indices) + "]"
    else:
        selector_slice = parse_slice_spec(spec["slice"], f"{context}.slice")
        indices = tuple(range(selector_slice.start, selector_slice.stop))
        selector_text = f"{selector_slice.start}:{selector_slice.stop}"

    return SourceSelection(
        source=source,
        indices=tuple(indices),
        field_names=field_names,
        selector_text=selector_text,
        label=str(spec.get("name")) if spec.get("name") else None,
        transform=parse_axis_transform(spec.get("transform", inherited_transform), context),
    )


def resolve_target_mapping(target, spec, info, global_transform=None):
    if not isinstance(spec, dict):
        raise ValueError(f"rmb_mappings.{target} must be an object.")

    inherited_source = spec.get("source")
    inherited_transform = spec.get("transform", global_transform)
    groups_cfg = spec.get("groups")
    if groups_cfg is None:
        groups = (resolve_selection_spec(spec, info, f"rmb_mappings.{target}", inherited_source=None, inherited_transform=global_transform),)
    else:
        if not isinstance(groups_cfg, list) or not groups_cfg:
            raise ValueError(f"rmb_mappings.{target}.groups must be a non-empty list.")
        groups = tuple(
            resolve_selection_spec(
                group_cfg,
                info,
                f"rmb_mappings.{target}.groups[{index}]",
                inherited_source=inherited_source,
                inherited_transform=inherited_transform,
            )
            for index, group_cfg in enumerate(groups_cfg)
        )

    operation = str(spec.get("operation", "concat")).strip()
    if operation not in {"concat", "xyz_pair_distance"}:
        raise ValueError(f"rmb_mappings.{target}.operation must be one of: concat, xyz_pair_distance.")
    if operation == "xyz_pair_distance":
        if len(groups) % 2 != 0:
            raise ValueError(f"rmb_mappings.{target}.operation=xyz_pair_distance requires an even number of groups.")
        for index, group in enumerate(groups):
            if group.dim != 3:
                raise ValueError(
                    f"rmb_mappings.{target}.groups[{index}] must select exactly x/y/z for xyz_pair_distance."
                )

    return TargetMapping(target=str(target), groups=groups, operation=operation)


def parse_default_target_policies(config):
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("default_targets in mapping config must be an object.")

    supported_targets = {"measured_joint_pos", "command_joint_pos"}
    supported_policies = {"auto", "remaining", "empty"}
    policies = {}
    for target, policy in config.items():
        target = str(target)
        policy = str(policy).strip().lower()
        if target not in supported_targets:
            raise ValueError(
                f"default_targets.{target} is not supported. "
                f"Supported targets: {', '.join(sorted(supported_targets))}"
            )
        if policy not in supported_policies:
            raise ValueError(
                f"default_targets.{target} must be one of: {', '.join(sorted(supported_policies))}"
            )
        policies[target] = policy
    return policies


def resolve_mapping_config(info, raw_mapping_config, mapping_config_path=None):
    raw_mapping_config = raw_mapping_config or {}
    vector_layout_overrides = parse_vector_layout_overrides(raw_mapping_config.get("vector_layout"))
    eef_transform = parse_axis_transform(raw_mapping_config.get("eef_transform"), "eef_transform")

    robot_layout_overrides = raw_mapping_config.get("robot_layout")
    if robot_layout_overrides is None:
        robot_layout_overrides = {}
    if not isinstance(robot_layout_overrides, dict):
        raise ValueError("robot_layout in mapping config must be an object.")

    target_mappings = {}
    mappings_cfg = raw_mapping_config.get("rmb_mappings")
    if mappings_cfg is not None:
        if not isinstance(mappings_cfg, dict):
            raise ValueError("rmb_mappings in mapping config must be an object.")
        for target, spec in mappings_cfg.items():
            target_name = str(target)
            target_mappings[target_name] = resolve_target_mapping(
                target_name,
                spec,
                info,
                global_transform=eef_transform if target_name in EEF_TARGETS else None,
            )

    return MappingConfig(
        path=str(Path(mapping_config_path).expanduser().resolve()) if mapping_config_path else None,
        vector_layout_overrides=vector_layout_overrides,
        robot_layout_overrides=robot_layout_overrides,
        default_target_policies=parse_default_target_policies(raw_mapping_config.get("default_targets")),
        target_mappings=target_mappings,
        eef_transform=eef_transform,
    )


def parse_range_slice(spec):
    if spec is None:
        return None
    try:
        return parse_slice_spec(spec, "range slice")
    except ValueError:
        return None


def slice_length(slice_obj):
    if slice_obj is None or slice_obj.stop is None or slice_obj.start is None:
        return None
    return max(0, int(slice_obj.stop) - int(slice_obj.start))


def resolve_vector_layout(info, modality, overrides=None):
    state_cfg = modality.get("state") if isinstance(modality, dict) else None
    action_cfg = modality.get("action") if isinstance(modality, dict) else None
    overrides = overrides or {}

    return VectorLayout(
        qpos_slice=overrides["qpos"] if "qpos" in overrides else (
            parse_range_slice(state_cfg.get("qpos")) if isinstance(state_cfg, dict) else None
        ),
        qvel_slice=overrides["qvel"] if "qvel" in overrides else (
            parse_range_slice(state_cfg.get("qvel")) if isinstance(state_cfg, dict) else None
        ),
        effort_slice=overrides["effort"] if "effort" in overrides else (
            parse_range_slice(state_cfg.get("effort")) if isinstance(state_cfg, dict) else None
        ),
        action_slice=overrides["action"] if "action" in overrides else (
            parse_range_slice(action_cfg.get("qpos")) if isinstance(action_cfg, dict) else None
        ),
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
    return get_feature_shape_dim(info, "action") or 0


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
    if isinstance(raw_value, (list, tuple)):
        items = raw_value
    else:
        items = str(raw_value).split(",")
    for item in items:
        item = str(item).strip()
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
    if raw_value is None:
        return tuple()
    if isinstance(raw_value, (list, tuple)):
        return tuple(sorted(parse_int_list(raw_value, arg_name)))
    text = str(raw_value).strip().lower()
    if text in {"", "none", "null", "no", "false"}:
        return tuple()
    return parse_int_list(raw_value, arg_name)


def parse_string_list(raw_value):
    if isinstance(raw_value, (list, tuple)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
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


def inspect_parquet_vector_dims(parquet_path):
    column_names = pq.ParquetFile(parquet_path).schema_arrow.names
    target_columns = [name for name in ("observation.state", "action") if name in column_names]
    if not target_columns:
        return tuple()

    table = pq.read_table(parquet_path, columns=target_columns).slice(0, 1)
    dims = []
    for column_name in target_columns:
        column = table[column_name]
        if len(column) == 0:
            continue
        value = column[0].as_py()
        if isinstance(value, list):
            dims.append((column_name, len(value)))
    return tuple(dims)


def summarize_names(names, max_items=8):
    if not isinstance(names, (list, tuple)) or not names:
        return None
    labels = [str(name) for name in names]
    if len(labels) <= max_items:
        return ", ".join(labels)
    return ", ".join(labels[:max_items]) + f", ... ({len(labels)} total)"


def summarize_field_descriptions(field_descriptions, max_items=8):
    if not isinstance(field_descriptions, dict) or not field_descriptions:
        return None

    rows = []
    for field_key, field_spec in field_descriptions.items():
        if not isinstance(field_spec, dict):
            continue
        indices = field_spec.get("indices")
        if not isinstance(indices, list) or not indices:
            continue
        try:
            index_text = ",".join(str(int(index)) for index in indices)
        except (TypeError, ValueError):
            continue
        label = field_spec.get("description") or str(field_key).split("/")[-1]
        rows.append((tuple(indices), f"[{index_text}] {label}"))

    if not rows:
        return None

    rows.sort(key=lambda item: item[0])
    labels = [label for _, label in rows]
    if len(labels) <= max_items:
        return "; ".join(labels)
    return "; ".join(labels[:max_items]) + f"; ... ({len(labels)} total)"


def feature_summary_lines(feature_key, feature_spec):
    if not isinstance(feature_spec, dict):
        return [f"{feature_key}"]

    lines = [
        f"{feature_key} (dtype={feature_spec.get('dtype', '?')}, shape={feature_spec.get('shape', '?')})"
    ]

    names_summary = summarize_names(feature_spec.get("names"))
    if names_summary is not None:
        lines.append(f"names: {names_summary}")

    field_summary = summarize_field_descriptions(feature_spec.get("field_descriptions"))
    if field_summary is not None:
        lines.append(f"fields: {field_summary}")

    video_info = feature_spec.get("video_info") or feature_spec.get("info")
    if isinstance(video_info, dict) and video_info:
        info_text = ", ".join(f"{key}={value}" for key, value in video_info.items())
        lines.append(f"video: {info_text}")

    return lines


def describe_selection(selection):
    if selection.field_names is not None:
        selector = summarize_names(selection.field_names, max_items=12) or ", ".join(selection.field_names)
    else:
        selector = selection.selector_text
    suffix = f" transform={selection.transform.text}" if selection.transform is not None else ""
    return f"{selection.source}[{selector}]{suffix}"


def describe_target_mapping(mapping):
    description = " + ".join(describe_selection(group) for group in mapping.groups)
    if mapping.operation != "concat":
        return f"{mapping.operation}({description})"
    return description


def get_target_mapping(mapping_config, target):
    if mapping_config is None:
        return None
    return mapping_config.target_mappings.get(target)


def get_target_policy(mapping_config, target, default="auto"):
    if mapping_config is None:
        return default
    return mapping_config.default_target_policies.get(target, default)


def get_feature_dim_or_zero(info, feature_key):
    return get_feature_shape_dim(info, feature_key) or 0


def mapped_indices_for_source(mapping_config, source, target_prefix):
    if mapping_config is None:
        return set()

    indices = set()
    for target, mapping in mapping_config.target_mappings.items():
        if not target.startswith(target_prefix):
            continue
        for group in mapping.groups:
            if group.source == source:
                indices.update(group.indices)
    return indices


def describe_default_target_policy(info, mapping_config, target, source, total_dim, target_prefix):
    policy = get_target_policy(mapping_config, target)
    if policy == "auto":
        return None

    used = mapped_indices_for_source(mapping_config, source, target_prefix)
    if policy == "remaining":
        remaining = [index for index in range(total_dim) if index not in used]
        return f"{source}[remaining: {len(remaining)} dims] -> {target} (default={policy})"
    if policy == "empty":
        return f"{source}[none] -> {target} (0 dims, default={policy})"
    return None


def list_unmapped_source_fields(info, mapping_config, source, target_prefix):
    if mapping_config is None or not mapping_config.target_mappings:
        return None
    total_dim = get_feature_dim_or_zero(info, source)
    if total_dim <= 0:
        return None

    used = mapped_indices_for_source(mapping_config, source, target_prefix)
    unmapped = [index for index in range(total_dim) if index not in used]
    if not unmapped:
        return None

    names = source_feature_names(info, source)
    if names is not None and len(names) == total_dim:
        labels = [str(names[index]) for index in unmapped]
    else:
        labels = [str(index) for index in unmapped]

    return summarize_names(labels, max_items=12) or ", ".join(labels)


def mapping_output_names(target):
    outputs = {
        "measured_joint_pos": ("measured_joint_pos", "measured_joint_pos_rel"),
        "command_joint_pos": ("command_joint_pos", "command_joint_pos_rel"),
        "measured_joint_pos_rel": ("measured_joint_pos_rel",),
        "command_joint_pos_rel": ("command_joint_pos_rel",),
        "measured_gripper_joint_pos": ("measured_gripper_joint_pos", "measured_gripper_joint_pos_rel"),
        "command_gripper_joint_pos": ("command_gripper_joint_pos", "command_gripper_joint_pos_rel"),
        "measured_gripper_joint_pos_rel": ("measured_gripper_joint_pos_rel",),
        "command_gripper_joint_pos_rel": ("command_gripper_joint_pos_rel",),
        "measured_eef_pose": ("measured_eef_pose", "measured_eef_pose_rel"),
        "command_eef_pose": ("command_eef_pose", "command_eef_pose_rel"),
        "measured_eef_pose_rel": ("measured_eef_pose_rel",),
        "command_eef_pose_rel": ("command_eef_pose_rel",),
        "measured_joint_vel": ("measured_joint_vel",),
        "measured_eef_wrench": ("measured_eef_wrench",),
        "time": ("time",),
        "task_desc": ("task_desc attribute",),
    }
    return outputs.get(target, (target,))


def format_mapping_block(title, details):
    lines = [title]
    for label, value in details:
        if value is None or value == "":
            continue
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def summarize_fk_mapping(robot_layout):
    fk_parts = []
    for arm_index, (arm_slice, fk_dim, target_link) in enumerate(
        zip(robot_layout.arm_slices, robot_layout.arm_fk_dims, robot_layout.eef_target_links)
    ):
        if fk_dim <= 0 or target_link is None:
            fk_parts.append(f"arm{arm_index}[{arm_slice.start}:{arm_slice.stop}] => zero EEF")
        else:
            fk_parts.append(
                f"arm{arm_index}[{arm_slice.start}:{arm_slice.stop}] fk={fk_dim} -> {target_link}"
            )
    return "; ".join(fk_parts)


def summarize_default_target(info, vector_layout, mapping_config, target, source, target_prefix, auto_selector, auto_dim):
    policy = get_target_policy(mapping_config, target)
    if policy == "remaining":
        total_dim = get_feature_dim_or_zero(info, source)
        used = mapped_indices_for_source(mapping_config, source, target_prefix)
        remaining = [index for index in range(total_dim) if index not in used]
        return f"{source}[remaining]", len(remaining), f"default={policy}"
    if policy == "empty":
        return f"{source}[none]", 0, f"default={policy}"
    return f"{source}[{auto_selector}]", auto_dim, "default=auto"


def effective_target_dim(info, vector_layout, mapping_config, target, source, target_prefix, auto_selector, auto_dim):
    target_mapping = get_target_mapping(mapping_config, target)
    if target_mapping is not None:
        return target_mapping.dim
    _, dim, _ = summarize_default_target(
        info,
        vector_layout,
        mapping_config,
        target=target,
        source=source,
        target_prefix=target_prefix,
        auto_selector=auto_selector,
        auto_dim=auto_dim,
    )
    return dim or 0


def build_mapping_lines(info, vector_layout, robot_layout, mapping_config=None):
    action_dim = resolve_action_dim(info, vector_layout)
    action_names = extract_action_names(info, vector_layout, action_dim) if action_dim is not None else None
    qpos_dim = slice_length(vector_layout.qpos_slice) or action_dim
    measured_joint_pos_dim = effective_target_dim(
        info,
        vector_layout,
        mapping_config,
        target="measured_joint_pos",
        source="observation.state",
        target_prefix="measured_",
        auto_selector=format_slice(vector_layout.qpos_slice),
        auto_dim=qpos_dim,
    )
    video_specs = list_video_specs(info)

    lines = []

    measured_joint_pos_mapping = get_target_mapping(mapping_config, "measured_joint_pos")
    if measured_joint_pos_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_joint_pos",
                [
                    ("source", describe_target_mapping(measured_joint_pos_mapping)),
                    ("dims", f"{measured_joint_pos_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("measured_joint_pos"))),
                ],
            )
        )
    else:
        source_text, dim_text, mode_text = summarize_default_target(
            info,
            vector_layout,
            mapping_config,
            target="measured_joint_pos",
            source="observation.state",
            target_prefix="measured_",
            auto_selector=format_slice(vector_layout.qpos_slice),
            auto_dim=qpos_dim,
        )
        lines.append(
            format_mapping_block(
                "measured_joint_pos",
                [
                    ("source", source_text),
                    ("dims", str(dim_text) if dim_text is not None else None),
                    ("mode", mode_text),
                    ("writes", ", ".join(mapping_output_names("measured_joint_pos"))),
                ],
            )
        )

    measured_joint_vel_mapping = get_target_mapping(mapping_config, "measured_joint_vel")
    if measured_joint_vel_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_joint_vel",
                [
                    ("source", describe_target_mapping(measured_joint_vel_mapping)),
                    ("dims", f"{measured_joint_vel_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("measured_joint_vel"))),
                ],
            )
        )
    else:
        if vector_layout.qvel_slice is not None:
            lines.append(
                format_mapping_block(
                    "measured_joint_vel",
                    [
                        ("source", f"observation.state[{format_slice(vector_layout.qvel_slice)}]"),
                        ("dims", str(slice_length(vector_layout.qvel_slice))),
                        ("mode", "default=auto"),
                        ("writes", ", ".join(mapping_output_names("measured_joint_vel"))),
                    ],
                )
            )
        else:
            lines.append(
                format_mapping_block(
                    "measured_joint_vel",
                    [
                        ("source", "none"),
                        ("dims", str(measured_joint_pos_dim)),
                        ("mode", "zero-filled when absent"),
                        ("writes", ", ".join(mapping_output_names("measured_joint_vel"))),
                    ],
                )
            )

    measured_wrench_mapping = get_target_mapping(mapping_config, "measured_eef_wrench")
    if measured_wrench_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_eef_wrench",
                [
                    ("source", describe_target_mapping(measured_wrench_mapping)),
                    ("dims", f"{measured_wrench_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("measured_eef_wrench"))),
                ],
            )
        )
    else:
        if vector_layout.effort_slice is not None:
            lines.append(
                format_mapping_block(
                    "measured_eef_wrench",
                    [
                        ("source", f"observation.state[{format_slice(vector_layout.effort_slice)}]"),
                        ("dims", f"{6 * len(robot_layout.arm_slices)} output dims"),
                        ("mode", "default=auto from effort; per arm first min(6, fk_dim) dims"),
                        ("writes", ", ".join(mapping_output_names("measured_eef_wrench"))),
                    ],
                )
            )
        else:
            lines.append(
                format_mapping_block(
                    "measured_eef_wrench",
                    [
                        ("source", "none"),
                        ("dims", f"{6 * len(robot_layout.arm_slices)} output dims"),
                        ("mode", "zero-filled when absent"),
                        ("writes", ", ".join(mapping_output_names("measured_eef_wrench"))),
                    ],
                )
            )

    command_joint_pos_mapping = get_target_mapping(mapping_config, "command_joint_pos")
    if command_joint_pos_mapping is not None:
        lines.append(
            format_mapping_block(
                "command_joint_pos",
                [
                    ("source", describe_target_mapping(command_joint_pos_mapping)),
                    ("dims", f"{command_joint_pos_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("command_joint_pos"))),
                ],
            )
        )
    else:
        source_text, dim_text, mode_text = summarize_default_target(
            info,
            vector_layout,
            mapping_config,
            target="command_joint_pos",
            source="action",
            target_prefix="command_",
            auto_selector=format_slice(vector_layout.action_slice),
            auto_dim=action_dim,
        )
        lines.append(
            format_mapping_block(
                "command_joint_pos",
                [
                    ("source", source_text),
                    ("dims", str(dim_text) if dim_text is not None else None),
                    ("mode", mode_text),
                    ("writes", ", ".join(mapping_output_names("command_joint_pos"))),
                ],
            )
        )

    measured_gripper_mapping = get_target_mapping(mapping_config, "measured_gripper_joint_pos")
    command_gripper_mapping = get_target_mapping(mapping_config, "command_gripper_joint_pos")
    measured_gripper_rel_mapping = get_target_mapping(mapping_config, "measured_gripper_joint_pos_rel")
    command_gripper_rel_mapping = get_target_mapping(mapping_config, "command_gripper_joint_pos_rel")
    if measured_gripper_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_gripper_joint_pos",
                [
                    ("source", describe_target_mapping(measured_gripper_mapping)),
                    ("dims", f"{measured_gripper_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("measured_gripper_joint_pos"))),
                ],
            )
        )
    if command_gripper_mapping is not None:
        lines.append(
            format_mapping_block(
                "command_gripper_joint_pos",
                [
                    ("source", describe_target_mapping(command_gripper_mapping)),
                    ("dims", f"{command_gripper_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("command_gripper_joint_pos"))),
                ],
            )
        )
    if measured_gripper_rel_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_gripper_joint_pos_rel",
                [
                    ("source", describe_target_mapping(measured_gripper_rel_mapping)),
                    ("dims", f"{measured_gripper_rel_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("measured_gripper_joint_pos_rel"))),
                ],
            )
        )
    if command_gripper_rel_mapping is not None:
        lines.append(
            format_mapping_block(
                "command_gripper_joint_pos_rel",
                [
                    ("source", describe_target_mapping(command_gripper_rel_mapping)),
                    ("dims", f"{command_gripper_rel_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("command_gripper_joint_pos_rel"))),
                ],
            )
        )
    if measured_gripper_mapping is None and command_gripper_mapping is None:
        if robot_layout.gripper_indices:
            lines.append(
                format_mapping_block(
                    "measured_gripper_joint_pos / command_gripper_joint_pos",
                    [
                        ("source", f"joint dims {robot_layout.gripper_indices} extracted from measured/command joint targets"),
                        ("dims", str(len(robot_layout.gripper_indices))),
                        ("mode", "default=auto"),
                        ("writes", ", ".join(mapping_output_names("measured_gripper_joint_pos") + mapping_output_names("command_gripper_joint_pos"))),
                    ],
                )
            )
        else:
            lines.append(
                format_mapping_block(
                    "measured_gripper_joint_pos / command_gripper_joint_pos",
                    [
                        ("source", "none"),
                        ("dims", "0"),
                        ("mode", "no gripper joints configured"),
                    ],
                )
            )

    measured_eef_mapping = get_target_mapping(mapping_config, "measured_eef_pose")
    command_eef_mapping = get_target_mapping(mapping_config, "command_eef_pose")
    measured_eef_rel_mapping = get_target_mapping(mapping_config, "measured_eef_pose_rel")
    command_eef_rel_mapping = get_target_mapping(mapping_config, "command_eef_pose_rel")
    if measured_eef_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_eef_pose",
                [
                    ("source", describe_target_mapping(measured_eef_mapping)),
                    ("dims", f"{measured_eef_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("measured_eef_pose"))),
                ],
            )
        )
    if command_eef_mapping is not None:
        lines.append(
            format_mapping_block(
                "command_eef_pose",
                [
                    ("source", describe_target_mapping(command_eef_mapping)),
                    ("dims", f"{command_eef_mapping.dim}"),
                    ("mode", "configured"),
                    ("writes", ", ".join(mapping_output_names("command_eef_pose"))),
                ],
            )
        )
    if measured_eef_rel_mapping is not None:
        lines.append(
            format_mapping_block(
                "measured_eef_pose_rel",
                [
                    ("source", describe_target_mapping(measured_eef_rel_mapping)),
                    ("dims", f"{measured_eef_rel_mapping.dim} input dims"),
                    ("mode", "configured; 7D relative pose groups are converted to 6D RMB deltas"),
                    ("writes", ", ".join(mapping_output_names("measured_eef_pose_rel"))),
                ],
            )
        )
    if command_eef_rel_mapping is not None:
        lines.append(
            format_mapping_block(
                "command_eef_pose_rel",
                [
                    ("source", describe_target_mapping(command_eef_rel_mapping)),
                    ("dims", f"{command_eef_rel_mapping.dim} input dims"),
                    ("mode", "configured; 7D relative pose groups are converted to 6D RMB deltas"),
                    ("writes", ", ".join(mapping_output_names("command_eef_pose_rel"))),
                ],
            )
        )
    if measured_eef_mapping is None and command_eef_mapping is None:
        lines.append(
            format_mapping_block(
                "measured_eef_pose / command_eef_pose",
                [
                    ("source", summarize_fk_mapping(robot_layout)),
                    ("dims", f"{7 * len(robot_layout.arm_slices)} output dims"),
                    ("mode", "default=FK / zero EEF when fk_dim<=0"),
                    ("writes", ", ".join(mapping_output_names("measured_eef_pose") + mapping_output_names("command_eef_pose"))),
                ],
            )
        )
    else:
        if measured_eef_mapping is None:
            lines.append(
                format_mapping_block(
                    "measured_eef_pose",
                    [
                        ("source", summarize_fk_mapping(robot_layout)),
                        ("dims", f"{7 * len(robot_layout.arm_slices)} output dims"),
                        ("mode", "default=FK / zero EEF when fk_dim<=0"),
                        ("writes", ", ".join(mapping_output_names("measured_eef_pose"))),
                    ],
                )
            )
        if command_eef_mapping is None:
            command_source = summarize_fk_mapping(robot_layout)
            command_mode = "default=FK / zero EEF when fk_dim<=0"
            if command_eef_rel_mapping is not None and measured_eef_mapping is not None:
                command_source = "measured_eef_pose + configured command_eef_pose_rel integration"
                command_mode = "derived from relative command"
            lines.append(
                format_mapping_block(
                    "command_eef_pose",
                    [
                        ("source", command_source),
                        ("dims", f"{7 * len(robot_layout.arm_slices)} output dims"),
                        ("mode", command_mode),
                        ("writes", ", ".join(mapping_output_names("command_eef_pose"))),
                    ],
                )
            )

    if "timestamp" in info_features(info):
        lines.append(
            format_mapping_block(
                "time",
                [
                    ("source", "timestamp"),
                    ("mode", "direct"),
                    ("writes", ", ".join(mapping_output_names("time"))),
                ],
            )
        )
    else:
        lines.append(
            format_mapping_block(
                "time",
                [
                    ("source", "timestamp missing"),
                    ("mode", "derived from fps and sample index"),
                    ("writes", ", ".join(mapping_output_names("time"))),
                ],
            )
        )

    lines.append(
        format_mapping_block(
            "task_desc",
            [
                ("source", "task_index + meta/tasks.jsonl + meta/episodes.jsonl"),
                ("mode", "episode attribute"),
                ("writes", ", ".join(mapping_output_names("task_desc"))),
            ],
        )
    )

    if video_specs:
        for spec in video_specs:
            suffix = "depth_image" if spec["is_depth"] else "rgb_image"
            lines.append(
                format_mapping_block(
                    f"{spec['camera_name']}_{suffix}.rmb.mp4",
                    [
                        ("source", spec["feature_key"]),
                        ("mode", "video export"),
                    ],
                )
            )
    else:
        lines.append(
            format_mapping_block(
                "video export",
                [
                    ("source", "observation.images.*"),
                    ("mode", "no video features found"),
                ],
            )
        )

    if action_names:
        lines.append(
            format_mapping_block(
                "action field order",
                [
                    ("fields", ", ".join(action_names)),
                    ("dims", str(len(action_names))),
                ],
            )
        )

    action_unmapped = list_unmapped_source_fields(info, mapping_config, "action", "command_")
    if action_unmapped:
        lines.append(
            format_mapping_block(
                "action unmapped fields",
                [
                    ("fields", action_unmapped),
                ],
            )
        )

    observation_unmapped = list_unmapped_source_fields(info, mapping_config, "observation.state", "measured_")
    if observation_unmapped:
        lines.append(
            format_mapping_block(
                "observation.state unmapped fields",
                [
                    ("fields", observation_unmapped),
                ],
            )
        )

    return lines


def print_dataset_mapping_summary(dataset_name, info, vector_layout, robot_layout, mapping_config=None, parquet_columns=None, parquet_vector_dims=None):
    features = []
    for feature_key, feature_spec in info_features(info).items():
        summary_lines = feature_summary_lines(feature_key, feature_spec)
        if isinstance(feature_spec, dict):
            video_info = feature_spec.get("video_info") or feature_spec.get("info")
            features.append(
                {
                    "key": feature_key,
                    "dtype": feature_spec.get("dtype"),
                    "shape": feature_spec.get("shape"),
                    "names": summarize_names(feature_spec.get("names")),
                    "fields": summarize_field_descriptions(feature_spec.get("field_descriptions")),
                    "video": video_info if isinstance(video_info, dict) and video_info else None,
                    "summary": summary_lines,
                }
            )
        else:
            features.append({"key": feature_key, "summary": summary_lines})

    mappings = []
    for block in build_mapping_lines(info, vector_layout, robot_layout, mapping_config=mapping_config):
        block_lines = block.splitlines()
        if not block_lines:
            continue
        mapping = {"target": block_lines[0]}
        for line in block_lines[1:]:
            label, sep, value = line.partition(":")
            if sep:
                mapping[label.strip()] = value.strip()
        mappings.append(mapping)

    summary = {
        "dataset": dataset_name,
        "lerobot_keys": features,
        "parquet_schema": [
            {"column": column_name, "type": str(column_type)}
            for column_name, column_type in (parquet_columns or [])
        ],
        "parquet_sample_dims": [
            {"column": column_name, "dim": dim}
            for column_name, dim in (parquet_vector_dims or [])
        ],
        "lerobot_to_rmb_mapping": mappings,
    }

    with open(f"{dataset_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def load_dataset_bundle(
    dataset_dir,
    robot_urdf,
    raw_mapping_config=None,
    mapping_config_path=None,
    arm_joint_dims=None,
    gripper_indices=None,
    arm_fk_dims=None,
    eef_target_links=None,
    robot_name=None,
):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_name = dataset_dir.name
    info, episodes_meta, tasks_by_index, modality = load_dataset_metadata(dataset_dir)
    mapping_config = resolve_mapping_config(info, raw_mapping_config, mapping_config_path=mapping_config_path)
    vector_layout = resolve_vector_layout(info, modality, overrides=mapping_config.vector_layout_overrides)
    robot_layout = resolve_robot_layout(
        info=info,
        robot_urdf=robot_urdf,
        vector_layout=vector_layout,
        arm_joint_dims=arm_joint_dims if arm_joint_dims is not None else mapping_config.robot_layout_overrides.get("arm_joint_dims"),
        gripper_indices=gripper_indices if gripper_indices is not None else mapping_config.robot_layout_overrides.get("gripper_indices"),
        arm_fk_dims=arm_fk_dims if arm_fk_dims is not None else mapping_config.robot_layout_overrides.get("arm_fk_dims"),
        eef_target_links=eef_target_links if eef_target_links is not None else mapping_config.robot_layout_overrides.get("eef_target_links"),
        robot_name=robot_name if robot_name is not None else mapping_config.robot_layout_overrides.get("robot_name"),
    )
    parquet_files = tuple(list_episode_parquet_files(dataset_dir))
    video_specs = tuple(list_video_specs(info))
    parquet_columns = list_parquet_columns(parquet_files[0]) if parquet_files else tuple()
    parquet_vector_dims = inspect_parquet_vector_dims(parquet_files[0]) if parquet_files else tuple()

    return DatasetBundle(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        info=info,
        episodes_meta=episodes_meta,
        tasks_by_index=tasks_by_index,
        vector_layout=vector_layout,
        robot_layout=robot_layout,
        mapping_config=mapping_config,
        parquet_files=parquet_files,
        parquet_columns=parquet_columns,
        parquet_vector_dims=parquet_vector_dims,
        video_specs=video_specs,
    )


def print_dataset_overview(bundle):
    print(f"\n📦 Processing dataset: {bundle.dataset_name}")
    if bundle.mapping_config.path is not None:
        print(f"🗺️  Mapping config: {bundle.mapping_config.path}")
    if bundle.mapping_config.eef_transform is not None:
        print(f"🧭 EEF transform: {bundle.mapping_config.eef_transform.text}")
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
        mapping_config=bundle.mapping_config,
        parquet_columns=bundle.parquet_columns,
        parquet_vector_dims=bundle.parquet_vector_dims,
    )


def build_episode_jobs(bundle, out_dir, requested_fps, camera_workers, video_preset, robot_urdf, skip_static_eef=False):
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
            mapping_config=bundle.mapping_config,
            video_specs=bundle.video_specs,
            skip_static_eef=skip_static_eef,
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


def compute_stacked_eef_pose_rel(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[1] == 0:
        return np.zeros((poses.shape[0], 0), dtype=np.float64)
    if poses.shape[1] % 7 != 0:
        raise ValueError(f"EEF pose array must have 7 columns per end effector, got shape {poses.shape}.")
    rel_parts = [
        compute_eef_pose_rel(poses[:, start : start + 7])
        for start in range(0, poses.shape[1], 7)
    ]
    return np.concatenate(rel_parts, axis=1)


def relative_pose7_to_delta6(relative_poses):
    relative_poses = np.asarray(relative_poses, dtype=np.float64)
    if relative_poses.shape[1] == 0:
        return np.zeros((relative_poses.shape[0], 0), dtype=np.float64)
    if relative_poses.shape[1] % 7 != 0:
        raise ValueError(f"Relative EEF pose source must have 7 columns per end effector, got shape {relative_poses.shape}.")

    rel_parts = []
    for start in range(0, relative_poses.shape[1], 7):
        block = relative_poses[:, start : start + 7]
        delta = np.zeros((relative_poses.shape[0], 6), dtype=np.float64)
        delta[:, :3] = block[:, :3]
        for index, quat in enumerate(block[:, 3:]):
            delta[index, 3:] = rotation_vector_from_matrix(quaternion_to_rotation_matrix(quat))
        rel_parts.append(delta)
    return np.concatenate(rel_parts, axis=1)


def axis_transform_matrix(transform):
    matrix = np.zeros((3, 3), dtype=np.float64)
    for output_axis, (source_axis, sign) in enumerate(zip(transform.perm, transform.signs)):
        matrix[output_axis, source_axis] = sign
    return matrix


def apply_axis_transform_vectors(values, transform):
    if transform is None or values.shape[1] == 0:
        return values
    return values[:, list(transform.perm)] * np.asarray(transform.signs, dtype=np.float64)


def apply_axis_transform_pose7(poses, transform):
    poses = np.asarray(poses, dtype=np.float64)
    if transform is None or poses.shape[1] == 0:
        return poses
    if poses.shape[1] % 7 != 0:
        raise ValueError(f"Expected 7D pose blocks for axis transform, got shape {poses.shape}.")

    matrix = axis_transform_matrix(transform)
    transformed = np.zeros_like(poses)
    for start in range(0, poses.shape[1], 7):
        block = poses[:, start : start + 7]
        transformed[:, start : start + 3] = apply_axis_transform_vectors(block[:, :3], transform)
        for index, quat in enumerate(block[:, 3:]):
            rotation = quaternion_to_rotation_matrix(quat)
            transformed[index, start + 3 : start + 7] = quaternion_xyzw_from_rotation(matrix @ rotation @ matrix.T)
    return transformed


def apply_axis_transform_delta6(deltas, transform):
    deltas = np.asarray(deltas, dtype=np.float64)
    if transform is None or deltas.shape[1] == 0:
        return deltas
    if deltas.shape[1] % 6 != 0:
        raise ValueError(f"Expected 6D delta blocks for axis transform, got shape {deltas.shape}.")

    transformed = np.zeros_like(deltas)
    for start in range(0, deltas.shape[1], 6):
        transformed[:, start : start + 3] = apply_axis_transform_vectors(deltas[:, start : start + 3], transform)
        transformed[:, start + 3 : start + 6] = apply_axis_transform_vectors(deltas[:, start + 3 : start + 6], transform)
    return transformed


def rotation_matrix_from_rotation_vector(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec)
    if np.isclose(theta, 0.0):
        return np.eye(3, dtype=np.float64)

    axis = rotvec / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def compose_pose_delta(pose, delta):
    pose = np.asarray(pose, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    base_rot = quaternion_to_rotation_matrix(pose[3:])
    delta_rot = rotation_matrix_from_rotation_vector(delta[3:])
    out = np.zeros(7, dtype=np.float64)
    out[:3] = pose[:3] + delta[:3]
    out[3:] = quaternion_xyzw_from_rotation(base_rot @ delta_rot)
    if np.dot(out[3:], pose[3:]) < 0.0:
        out[3:] *= -1.0
    return out


def integrate_stacked_eef_pose_rel(initial_poses, rel):
    initial_poses = np.asarray(initial_poses, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    if rel.shape[1] == 0:
        return np.zeros((rel.shape[0], 0), dtype=np.float64)
    if initial_poses.shape[1] % 7 != 0 or rel.shape[1] % 6 != 0:
        raise ValueError(
            f"Expected 7D initial poses and 6D relative deltas, got initial={initial_poses.shape}, rel={rel.shape}."
        )
    if initial_poses.shape[1] // 7 != rel.shape[1] // 6:
        raise ValueError(
            f"Initial pose EEF count and relative EEF count differ: {initial_poses.shape[1] // 7} vs {rel.shape[1] // 6}."
        )

    integrated = np.zeros((rel.shape[0], initial_poses.shape[1]), dtype=np.float64)
    if len(rel) == 0:
        return integrated

    integrated[0] = initial_poses[0]
    for row in range(len(rel)):
        if row > 0:
            integrated[row] = integrated[row - 1]
        for pose_start, rel_start in zip(range(0, initial_poses.shape[1], 7), range(0, rel.shape[1], 6)):
            integrated[row, pose_start : pose_start + 7] = compose_pose_delta(
                integrated[row, pose_start : pose_start + 7],
                rel[row, rel_start : rel_start + 6],
            )
    return integrated


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


def extract_index_columns(values, indices, label):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {values.shape}.")
    if not indices:
        return np.zeros((values.shape[0], 0), dtype=np.float64)
    max_index = max(indices)
    if max_index >= values.shape[1]:
        raise ValueError(
            f"{label} refers to index {max_index}, but the source only has {values.shape[1]} columns."
        )
    return values[:, indices]


def extract_target_mapping_array(mapping, source_arrays):
    missing_sources = sorted({group.source for group in mapping.groups if group.source not in source_arrays})
    if missing_sources:
        return None

    chunks = []
    num_steps = None
    for group in mapping.groups:
        source_array = np.asarray(source_arrays[group.source], dtype=np.float64)
        if source_array.ndim != 2:
            raise ValueError(f"Mapping source {group.source!r} must be 2D, got shape {source_array.shape}.")
        if num_steps is None:
            num_steps = source_array.shape[0]
        elif source_array.shape[0] != num_steps:
            raise ValueError(f"All mapping sources for {mapping.target} must have the same step count.")
        chunks.append(extract_index_columns(source_array, group.indices, f"{mapping.target}:{group.source}"))

    if num_steps is None:
        return np.zeros((0, 0), dtype=np.float64)

    if mapping.target in {"measured_eef_pose", "command_eef_pose"}:
        transformed_chunks = []
        for chunk, group in zip(chunks, mapping.groups):
            if chunk.shape[1] != 7:
                raise ValueError(f"{mapping.target} requires 7 dims per group, got {chunk.shape[1]}.")
            transformed_chunks.append(apply_axis_transform_pose7(chunk, group.transform))
        chunks = transformed_chunks
    elif mapping.target in {"measured_eef_pose_rel", "command_eef_pose_rel"}:
        normalized = []
        for chunk, group in zip(chunks, mapping.groups):
            if chunk.shape[1] == 6:
                normalized.append(apply_axis_transform_delta6(chunk, group.transform))
            elif chunk.shape[1] == 7:
                normalized.append(apply_axis_transform_delta6(relative_pose7_to_delta6(chunk), group.transform))
            else:
                raise ValueError(f"{mapping.target} requires 6D deltas or 7D relative poses per group, got {chunk.shape[1]}.")
        chunks = normalized
    elif mapping.target == "measured_eef_wrench":
        padded = []
        for chunk in chunks:
            if chunk.shape[1] > 6:
                raise ValueError(f"{mapping.target} allows at most 6 dims per group, got {chunk.shape[1]}.")
            if chunk.shape[1] < 6:
                pad = np.zeros((num_steps, 6 - chunk.shape[1]), dtype=np.float64)
                chunk = np.concatenate([chunk, pad], axis=1)
            padded.append(chunk)
        chunks = padded

    if mapping.operation == "xyz_pair_distance":
        chunks = [
            np.linalg.norm(chunks[index] - chunks[index + 1], axis=1, keepdims=True)
            for index in range(0, len(chunks), 2)
        ]

    return np.concatenate(chunks, axis=1) if chunks else np.zeros((num_steps, 0), dtype=np.float64)


def build_default_target_override(mapping_config, target, source_arrays, source_name, target_prefix):
    policy = get_target_policy(mapping_config, target)
    if policy == "auto":
        return None

    source_array = np.asarray(source_arrays[source_name], dtype=np.float64)
    num_steps = source_array.shape[0]
    if policy == "empty":
        return np.zeros((num_steps, 0), dtype=np.float64)
    if policy == "remaining":
        used = mapped_indices_for_source(mapping_config, source_name, target_prefix)
        remaining = [index for index in range(source_array.shape[1]) if index not in used]
        return source_array[:, remaining]
    raise ValueError(f"Unsupported default target policy for {target}: {policy}")


def resolve_target_overrides(mapping_config, source_arrays):
    overrides = {}
    if mapping_config is None:
        return overrides

    for target, mapping in mapping_config.target_mappings.items():
        value = extract_target_mapping_array(mapping, source_arrays)
        if value is not None:
            overrides[target] = value

    default_specs = (
        ("measured_joint_pos", "observation.state", "measured_"),
        ("command_joint_pos", "action", "command_"),
    )
    for target, source_name, target_prefix in default_specs:
        if target in overrides:
            continue
        default_value = build_default_target_override(mapping_config, target, source_arrays, source_name, target_prefix)
        if default_value is not None:
            overrides[target] = default_value

    return overrides


def is_effectively_zero(values, atol=1e-12):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return True
    return not np.count_nonzero(np.abs(values) > atol)


def static_mapped_eef_reasons(target_overrides):
    reasons = []

    measured_eef_pose = target_overrides.get("measured_eef_pose")
    if measured_eef_pose is not None:
        measured_eef_pose_rel = compute_stacked_eef_pose_rel(measured_eef_pose)
        if is_effectively_zero(measured_eef_pose_rel):
            reasons.append("configured measured_eef_pose is static; measured_eef_pose_rel would be all zeros")

    measured_eef_pose_rel = target_overrides.get("measured_eef_pose_rel")
    if measured_eef_pose_rel is not None and is_effectively_zero(measured_eef_pose_rel):
        reasons.append("configured measured_eef_pose_rel is all zeros")

    command_eef_pose = target_overrides.get("command_eef_pose")
    if command_eef_pose is not None:
        command_eef_pose_rel = compute_stacked_eef_pose_rel(command_eef_pose)
        if is_effectively_zero(command_eef_pose_rel):
            reasons.append("configured command_eef_pose is static; command_eef_pose_rel would be all zeros")

    command_eef_pose_rel = target_overrides.get("command_eef_pose_rel")
    if command_eef_pose_rel is not None and is_effectively_zero(command_eef_pose_rel):
        reasons.append("configured command_eef_pose_rel is all zeros")

    return reasons


def warn_if_static_mapped_eef(parquet_path, target_overrides):
    for reason in static_mapped_eef_reasons(target_overrides):
        print(
            f"⚠️  Static EEF input in {parquet_path.name}: {reason}."
        )


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
    target_overrides=None,
):
    dtype_f8 = np.float64
    num_steps = len(time_values)
    target_overrides = target_overrides or {}

    measured_joint_pos = np.asarray(target_overrides.get("measured_joint_pos", qpos), dtype=dtype_f8)
    measured_joint_vel = np.asarray(target_overrides.get("measured_joint_vel", qvel), dtype=dtype_f8)
    command_joint_pos = np.asarray(target_overrides.get("command_joint_pos", action), dtype=dtype_f8)

    if measured_joint_vel.shape[1] != measured_joint_pos.shape[1]:
        if "measured_joint_vel" in target_overrides:
            raise ValueError(
                "Configured measured_joint_vel dims do not match measured_joint_pos: "
                f"{measured_joint_vel.shape[1]} vs {measured_joint_pos.shape[1]}"
            )
        measured_joint_vel = np.zeros_like(measured_joint_pos)

    if "measured_gripper_joint_pos" in target_overrides:
        measured_gripper_joint_pos = np.asarray(target_overrides["measured_gripper_joint_pos"], dtype=dtype_f8)
    elif layout.gripper_indices:
        measured_gripper_joint_pos = extract_index_columns(
            measured_joint_pos,
            layout.gripper_indices,
            "measured_gripper_joint_pos",
        )
    else:
        measured_gripper_joint_pos = np.zeros((num_steps, 0), dtype=dtype_f8)

    if "command_gripper_joint_pos" in target_overrides:
        command_gripper_joint_pos = np.asarray(target_overrides["command_gripper_joint_pos"], dtype=dtype_f8)
    elif layout.gripper_indices:
        command_gripper_joint_pos = extract_index_columns(
            command_joint_pos,
            layout.gripper_indices,
            "command_gripper_joint_pos",
        )
    else:
        command_gripper_joint_pos = np.zeros((num_steps, 0), dtype=dtype_f8)

    if "measured_eef_pose" in target_overrides:
        measured_eef_pose = np.asarray(target_overrides["measured_eef_pose"], dtype=dtype_f8)
    else:
        if measured_joint_pos.shape[1] < layout.total_arm_joint_dim:
            raise ValueError(
                f"Expected at least {layout.total_arm_joint_dim} measured joint dims for FK, got {measured_joint_pos.shape[1]}. "
                "Provide rmb_mappings.measured_eef_pose to bypass FK."
            )
        measured_eef_pose, _ = build_multiarm_eef_data(measured_joint_pos, kinematics_per_arm, layout)

    if "command_eef_pose" in target_overrides:
        command_eef_pose = np.asarray(target_overrides["command_eef_pose"], dtype=dtype_f8)
    elif "command_eef_pose_rel" in target_overrides and "measured_eef_pose" in target_overrides:
        command_eef_pose = integrate_stacked_eef_pose_rel(
            measured_eef_pose,
            np.asarray(target_overrides["command_eef_pose_rel"], dtype=dtype_f8),
        )
    elif "measured_eef_pose" in target_overrides:
        command_eef_pose = measured_eef_pose.copy()
    else:
        if command_joint_pos.shape[1] < layout.total_arm_joint_dim:
            raise ValueError(
                f"Expected at least {layout.total_arm_joint_dim} command joint dims for FK, got {command_joint_pos.shape[1]}. "
                "Provide rmb_mappings.command_eef_pose or rmb_mappings.command_eef_pose_rel to bypass FK."
            )
        command_eef_pose, _ = build_multiarm_eef_data(command_joint_pos, kinematics_per_arm, layout)

    measured_eef_pose_rel = compute_stacked_eef_pose_rel(measured_eef_pose)
    if "measured_eef_pose_rel" in target_overrides:
        measured_eef_pose_rel = np.asarray(target_overrides["measured_eef_pose_rel"], dtype=dtype_f8)
    if "command_eef_pose_rel" in target_overrides:
        command_eef_pose_rel = np.asarray(target_overrides["command_eef_pose_rel"], dtype=dtype_f8)
    else:
        command_eef_pose_rel = compute_stacked_eef_pose_rel(command_eef_pose)
    num_eef = measured_eef_pose.shape[1] // 7 if measured_eef_pose.shape[1] else (command_eef_pose.shape[1] // 7)
    if measured_eef_pose.shape[1] % 7 != 0 or command_eef_pose.shape[1] % 7 != 0:
        raise ValueError(
            f"EEF pose dims must be divisible by 7, got measured={measured_eef_pose.shape[1]}, command={command_eef_pose.shape[1]}"
        )
    if measured_eef_pose.shape[1] != command_eef_pose.shape[1]:
        raise ValueError(
            f"Measured and command EEF pose dims must match, got {measured_eef_pose.shape[1]} and {command_eef_pose.shape[1]}"
        )
    if measured_eef_pose_rel.shape[1] != 6 * num_eef or command_eef_pose_rel.shape[1] != 6 * num_eef:
        raise ValueError(
            f"EEF pose rel datasets must have 6 dims per end effector ({6 * num_eef} total), "
            f"got measured={measured_eef_pose_rel.shape[1]}, command={command_eef_pose_rel.shape[1]}"
        )

    if "measured_eef_wrench" in target_overrides:
        measured_eef_wrench = np.asarray(target_overrides["measured_eef_wrench"], dtype=dtype_f8)
    elif effort is not None:
        effort = np.asarray(effort, dtype=dtype_f8)
        if effort.shape[1] < layout.total_arm_joint_dim:
            raise ValueError(
                f"Expected at least {layout.total_arm_joint_dim} effort dimensions for configured robot layout, got {effort.shape[1]}"
            )
        measured_eef_wrench = build_multiarm_eef_wrench_data(effort, layout)
    else:
        measured_eef_wrench = np.zeros((num_steps, 6 * num_eef), dtype=dtype_f8)

    if measured_eef_wrench.shape[1] != 6 * num_eef:
        raise ValueError(
            f"measured_eef_wrench must have 6 dims per end effector ({6 * num_eef} total), got {measured_eef_wrench.shape[1]}"
        )

    def resolve_rel_dataset(target, default, reference):
        rel = np.asarray(target_overrides.get(target, default), dtype=dtype_f8)
        if rel.shape != reference.shape:
            raise ValueError(
                f"{target} must have shape {reference.shape}, got {rel.shape}."
            )
        return rel

    measured_joint_pos_rel = resolve_rel_dataset(
        "measured_joint_pos_rel",
        previous_step_delta(measured_joint_pos),
        measured_joint_pos,
    )
    command_joint_pos_rel = resolve_rel_dataset(
        "command_joint_pos_rel",
        previous_step_delta(command_joint_pos),
        command_joint_pos,
    )
    measured_gripper_joint_pos_rel = resolve_rel_dataset(
        "measured_gripper_joint_pos_rel",
        previous_step_delta(measured_gripper_joint_pos),
        measured_gripper_joint_pos,
    )
    command_gripper_joint_pos_rel = resolve_rel_dataset(
        "command_gripper_joint_pos_rel",
        previous_step_delta(command_gripper_joint_pos),
        command_gripper_joint_pos,
    )
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


def load_episode_from_parquet(parquet_path, mapping_config=None):
    schema = pq.ParquetFile(parquet_path).schema_arrow.names
    observation_column = resolve_column_name(schema, exact="observation.state", prefix="observation.state.")
    action_column = resolve_optional_column_name(schema, exact="action", prefix="action.")

    optional_columns = [name for name in ("timestamp", "task_index") if name in schema]
    source_columns = [observation_column] + optional_columns
    if action_column is not None:
        source_columns.append(action_column)
    for source in sorted(mapped_source_names(mapping_config)):
        if source in schema:
            source_columns.append(source)
        else:
            print(
                f"⚠️  Missing configured mapping source column '{source}' in {parquet_path.name}; "
                "dependent RMB targets will use fallback values when available."
            )
    source_columns = list(dict.fromkeys(source_columns))

    table = pq.read_table(parquet_path, columns=source_columns)

    observation_state = stack_list_column(table[observation_column], dtype=np.float64)
    if action_column is not None:
        action = stack_list_column(table[action_column], dtype=np.float64)
    else:
        action = np.zeros((observation_state.shape[0], 0), dtype=np.float64)

    timestamps = None
    if "timestamp" in table.column_names:
        timestamps = scalar_column_to_numpy(table["timestamp"], dtype=np.float64)

    task_indices = None
    if "task_index" in table.column_names:
        task_indices = scalar_column_to_numpy(table["task_index"], dtype=np.int64)

    source_arrays = {
        "observation.state": observation_state,
        "action": action,
    }
    for source in mapped_source_names(mapping_config):
        if source in table.column_names:
            source_arrays[source] = stack_list_column(table[source], dtype=np.float64)

    return observation_state, action, timestamps, task_indices, source_arrays


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

    print(f"📄 Reading parquet: {job.parquet_path}")

    observation_state, raw_action, timestamps, task_indices, source_arrays = load_episode_from_parquet(
        job.parquet_path,
        mapping_config=job.mapping_config,
    )
    action = apply_action_slice(raw_action, job.vector_layout, job.parquet_path)
    if action.shape[1] != job.robot_layout.total_arm_joint_dim:
        if "command_eef_pose" in job.mapping_config.target_mappings:
            print(
                f"ℹ️  action dim for {job.parquet_path.name} is {action.shape[1]}, "
                f"while the inferred robot layout uses {job.robot_layout.total_arm_joint_dim} dims for FK. "
                "Configured command_eef_pose mapping will bypass FK for the command pose."
            )
        else:
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
    source_arrays_rs = {
        source: values[sample_indices]
        for source, values in source_arrays.items()
    }
    target_overrides = resolve_target_overrides(job.mapping_config, source_arrays_rs)
    static_eef_reasons = static_mapped_eef_reasons(target_overrides)
    if static_eef_reasons:
        for reason in static_eef_reasons:
            prefix = "❌" if job.skip_static_eef else "⚠️"
            print(f"{prefix} Static EEF input in {job.parquet_path.name}: {reason}.")
        if job.skip_static_eef:
            if rmb_dir.exists():
                shutil.rmtree(rmb_dir)
            print(f"⏭️  Skipped static EEF episode: {episode_name}")
            return

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

    rmb_dir.mkdir(parents=True, exist_ok=True)
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
        target_overrides=target_overrides,
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
    mapping_config_path=None,
    eef_transform_axes=None,
    skip_static_eef=False,
    assume_yes=False,
):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_urdf = resolve_robot_urdf(robot_urdf)
    raw_mapping_config = read_optional_json(mapping_config_path)
    eef_transform = parse_eef_transform_axes_arg(eef_transform_axes)
    if eef_transform is not None:
        raw_mapping_config = dict(raw_mapping_config)
        raw_mapping_config["eef_transform"] = eef_transform

    dataset_dirs = discover_lerobot_datasets(input_dir)
    if not dataset_dirs:
        print(f"❌ No LeRobot datasets found under: {input_dir}")
        return

    for dataset_dir in dataset_dirs:
        bundle = load_dataset_bundle(
            dataset_dir=dataset_dir,
            robot_urdf=robot_urdf,
            raw_mapping_config=raw_mapping_config,
            mapping_config_path=mapping_config_path,
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
        jobs = build_episode_jobs(
            bundle,
            out_dir,
            fps,
            camera_workers,
            video_preset,
            robot_urdf,
            skip_static_eef=skip_static_eef,
        )

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
        "--mapping_config",
        type=str,
        default=None,
        help="JSON file that explicitly maps LeRobot vectors into RMB datasets. Fields are resolved against raw LeRobot vectors before modality slicing.",
    )
    parser.add_argument(
        "--eef_transform_axes",
        type=str,
        default=None,
        help=(
            "Optional global signed axis transform for all EEF pose/relative mappings, "
            "for example 'x,-z,y' or '--eef_transform_axes=-x,y,z'. "
            "Overrides top-level eef_transform in --mapping_config. "
            "Group-level transform entries still take precedence."
        ),
    )
    parser.add_argument(
        "--skip_static_eef",
        action="store_true",
        help="Skip episodes whose configured EEF pose/relative sources are static, and remove any existing output for those episodes.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the y/n confirmation prompt after showing the LeRobot keys and RMB mapping.",
    )
    args = parser.parse_args(normalize_cli_args(sys.argv[1:]))

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
        mapping_config_path=args.mapping_config,
        eef_transform_axes=args.eef_transform_axes,
        skip_static_eef=args.skip_static_eef,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    main()
