#!/usr/bin/env python3

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:
    raise SystemExit("h5py is required to read RMB episodes.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_to_rmb import (
    ARM_DOF_FOR_FK,
    ARM_SLICES,
    UrdfKinematics,
    quaternion_xyzw_from_rotation,
    quaternion_to_rotation_matrix,
    rotation_vector_from_matrix,
    resolve_robot_urdf,
)


ARM_LABELS = (
    ("left", "#1f77b4", np.array([0.0, 0.30, 0.0], dtype=np.float64)),
    ("right", "#ff7f0e", np.array([0.0, -0.30, 0.0], dtype=np.float64)),
)
EEF_SLICES = (
    slice(0, 7),
    slice(7, 14),
)
EEF_REL_SLICES = (
    slice(0, 6),
    slice(6, 12),
)
JOINT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")
EEF_AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


@dataclass(frozen=True)
class EpisodeSource:
    dataset_name: str
    episode_name: str
    episode_dir: Path
    hdf5_path: Path


@dataclass(frozen=True)
class EpisodeVideoStream:
    name: str
    path: Path


@dataclass
class EpisodeVideoCapture:
    stream: EpisodeVideoStream
    capture: object
    next_frame_idx: int = 0


@dataclass
class EpisodeData:
    source: EpisodeSource
    task_desc: str
    time_values: np.ndarray
    measured_joint_pos: np.ndarray
    command_joint_pos: np.ndarray
    measured_eef_pose: np.ndarray
    command_eef_pose: np.ndarray
    measured_eef_pose_rel: np.ndarray
    command_eef_pose_rel: np.ndarray
    video_streams: tuple[EpisodeVideoStream, ...]


@dataclass
class EpisodeIkSolution:
    frame_indices: np.ndarray
    frame_index_to_position: dict[int, int]
    joints_by_source: dict[str, np.ndarray]
    position_error_by_source: dict[str, np.ndarray]
    rotation_error_by_source: dict[str, np.ndarray]
    iterations_by_source: dict[str, np.ndarray]


def has_slice(array, slice_obj):
    array = np.asarray(array)
    return array.ndim == 2 and slice_obj.stop <= array.shape[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Render an RMB episode or dataset preview into a single MP4.")
    parser.add_argument(
        "--input_dir",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more RMB paths. Directories are searched recursively for episode directories "
            "that contain main.rmb.hdf5; a main.rmb.hdf5 file can also be passed directly."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. The output MP4 file name is generated automatically.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional exact output MP4 path. Overrides --output_dir.",
    )
    parser.add_argument(
        "--robot_urdf",
        type=str,
        default=None,
        help="Path to the VX300S URDF. Defaults to config/assets vx300s.urdf.",
    )
    parser.add_argument(
        "--source",
        choices=("measured", "command", "both"),
        default="both",
        help="Which RMB source to draw.",
    )
    parser.add_argument(
        "--eef_mode",
        choices=("absolute", "relative", "auto"),
        default="absolute",
        help=(
            "EEF trajectory source. absolute draws *_eef_pose. relative integrates *_eef_pose_rel "
            "from an ALOHA/URDF initial pose. auto uses relative when rel datasets exist."
        ),
    )
    parser.add_argument(
        "--relative_origin",
        choices=("urdf_zero", "absolute_first", "joint_first"),
        default="urdf_zero",
        help=(
            "Initial pose for --eef_mode relative. urdf_zero uses all-zero ALOHA FK; "
            "absolute_first uses the first absolute EEF pose; joint_first uses first-frame FK when joint data exists."
        ),
    )
    parser.add_argument(
        "--hide_urdf_reference",
        action="store_true",
        help="Hide the faint all-zero URDF arm skeleton drawn in relative EEF mode.",
    )
    parser.add_argument(
        "--ik_preview",
        action="store_true",
        help="Run a DLS IK preview for the rendered EEF targets and draw the reconstructed arm joints.",
    )
    parser.add_argument(
        "--ik_seed",
        choices=("zero", "joint_first"),
        default="zero",
        help="Initial IK seed for the first rendered frame. Later frames start from the previous IK solution.",
    )
    parser.add_argument("--ik_max_iterations", type=int, default=45, help="Maximum IK iterations per rendered frame.")
    parser.add_argument("--ik_tolerance", type=float, default=1e-4, help="Weighted IK error tolerance.")
    parser.add_argument("--ik_damping", type=float, default=0.04, help="Damping factor for damped least-squares IK.")
    parser.add_argument("--ik_max_step", type=float, default=0.18, help="Maximum joint update norm per IK iteration.")
    parser.add_argument(
        "--ik_jacobian",
        choices=("geometric", "numerical"),
        default="geometric",
        help="Jacobian backend for IK. geometric is much faster; numerical is mainly for debugging.",
    )
    parser.add_argument("--ik_fd_eps", type=float, default=1e-5, help="Finite-difference epsilon for numerical IK Jacobian.")
    parser.add_argument("--ik_position_weight", type=float, default=1.0, help="IK weight for position error.")
    parser.add_argument("--ik_orientation_weight", type=float, default=0.35, help="IK weight for orientation error.")
    parser.add_argument(
        "--figure_scale",
        type=float,
        default=0.75,
        help="Scale factor for the rendered matplotlib canvas. Use 1.0 for the previous full-resolution output.",
    )
    parser.add_argument("--no_camera", action="store_true", help="Skip camera video panels for faster rendering.")
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="both",
        help="Which arm to draw.",
    )
    parser.add_argument("--frame_start", type=int, default=0, help="Start frame index for each episode.")
    parser.add_argument("--frame_stop", type=int, default=None, help="Exclusive stop frame index for each episode.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame stride for rendering.")
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=None,
        help="Optional maximum number of rendered frames per episode. When set, frames are sampled uniformly.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="Maximum number of episodes to include per discovered dataset directory. Use 0 or a negative value for no limit.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to the episode time delta and frame_stride.",
    )
    parser.add_argument(
        "--eef_axis_length",
        type=float,
        default=0.05,
        help="Length in meters for the current EEF orientation axes.",
    )
    parser.add_argument(
        "--bounds_samples",
        type=int,
        default=160,
        help="Number of sampled frames used to stabilize the 3D axis range per episode.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="Agg",
        help="Matplotlib backend. Defaults to Agg for headless MP4 export.",
    )
    return parser.parse_args()


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def discover_episode_sources(input_dir, max_episodes):
    input_dir = Path(input_dir).expanduser().resolve()
    if input_dir.is_file():
        if input_dir.name != "main.rmb.hdf5":
            raise FileNotFoundError(f"RMB file input must be named main.rmb.hdf5, got: {input_dir}")
        episode_dir = input_dir.parent.resolve()
        dataset_dir = episode_dir.parent.resolve()
        return [
            EpisodeSource(
                dataset_name=dataset_dir.name,
                episode_name=episode_dir.name,
                episode_dir=episode_dir,
                hdf5_path=input_dir,
            )
        ]

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input_dir expects a directory or main.rmb.hdf5 file, got: {input_dir}")

    episode_map = {}
    for hdf5_path in sorted(input_dir.rglob("main.rmb.hdf5")):
        episode_dir = hdf5_path.parent.resolve()
        dataset_dir = episode_dir.parent.resolve()
        episode_map.setdefault(dataset_dir, []).append(
            EpisodeSource(
                dataset_name=dataset_dir.name,
                episode_name=episode_dir.name,
                episode_dir=episode_dir,
                hdf5_path=hdf5_path.resolve(),
            )
        )

    if not episode_map:
        raise FileNotFoundError(f"No main.rmb.hdf5 files were found under: {input_dir}")

    episode_sources = []
    for dataset_dir in sorted(episode_map):
        sources = sorted(episode_map[dataset_dir], key=lambda item: item.episode_name)
        if max_episodes is not None and max_episodes > 0:
            sources = sources[:max_episodes]
        episode_sources.extend(sources)

    return episode_sources


def resolve_multiple_episode_sources(input_dirs, max_episodes):
    episode_sources = []
    seen_episode_dirs = set()
    for input_dir in input_dirs:
        for source in discover_episode_sources(input_dir, max_episodes):
            episode_key = str(source.episode_dir.resolve())
            if episode_key in seen_episode_dirs:
                continue
            seen_episode_dirs.add(episode_key)
            episode_sources.append(source)

    if not episode_sources:
        raise FileNotFoundError("No RMB episodes were found in the provided --input_dir values.")
    return episode_sources


def sanitize_filename_component(text):
    text = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text).strip())
    text = text.strip("_")
    return text or "preview"


def build_default_output_filename(input_dirs, episode_sources):
    resolved_inputs = [Path(input_dir).expanduser().resolve() for input_dir in input_dirs]
    if len(input_dirs) == 1 and len(episode_sources) == 1:
        source = episode_sources[0]
        return (
            f"{sanitize_filename_component(source.dataset_name)}__"
            f"{sanitize_filename_component(source.episode_name)}__preview.mp4"
        )
    if len(input_dirs) == 1:
        return f"{sanitize_filename_component(resolved_inputs[0].name)}__rmb_preview.mp4"

    input_names = [sanitize_filename_component(path.name) for path in resolved_inputs[:3]]
    if len(resolved_inputs) > 3:
        input_names.append(f"plus{len(resolved_inputs) - 3}")
    return "__".join(input_names) + "__rmb_preview.mp4"


def build_default_output_dir(input_dirs, episode_sources):
    if len(input_dirs) == 1 and len(episode_sources) == 1:
        return episode_sources[0].episode_dir
    if len(input_dirs) == 1:
        return Path(input_dirs[0]).expanduser().resolve()
    return Path.cwd()


def ensure_unique_output_path(output_path):
    output_path = Path(output_path).expanduser().resolve()
    if not output_path.exists():
        return output_path

    parent = output_path.parent
    stem = output_path.stem
    suffix = output_path.suffix
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_output_path(output_dir_arg, output_path_arg, input_dirs, episode_sources):
    if output_path_arg is not None:
        output_path = Path(output_path_arg).expanduser().resolve()
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
        return output_path

    output_filename = build_default_output_filename(input_dirs, episode_sources)
    if output_dir_arg is None:
        output_dir = build_default_output_dir(input_dirs, episode_sources)
    else:
        output_dir = Path(output_dir_arg).expanduser().resolve()
    return ensure_unique_output_path(output_dir / output_filename)


def load_matplotlib(backend):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for visualization. Install it with `pip install matplotlib`.") from exc

    matplotlib.use(backend)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to initialize matplotlib backend '{backend}': {exc}") from exc

    return plt


def load_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit("opencv-python is required for MP4 read/write in visualization.") from exc
    return cv2


def discover_episode_video_streams(episode_dir):
    streams = []
    for path in sorted(episode_dir.glob("*_rgb_image.rmb.mp4")):
        name = path.name.replace("_rgb_image.rmb.mp4", "")
        streams.append(EpisodeVideoStream(name=name, path=path))
    return tuple(streams)


def read_dataset_or_empty(root, key, num_frames):
    if key in root:
        return root[key][()]
    return np.zeros((num_frames, 0), dtype=np.float64)


def load_episode_data(source):
    with h5py.File(source.hdf5_path, "r") as f:
        task_desc = decode_attr(f.attrs.get("task_desc", source.dataset_name.replace("_", " ")))
        time_values = f["time"][()] if "time" in f else np.arange(len(f["measured_joint_pos"]), dtype=np.float64)
        num_frames = len(time_values)
        measured_joint_pos = read_dataset_or_empty(f, "measured_joint_pos", num_frames)
        command_joint_pos = read_dataset_or_empty(f, "command_joint_pos", num_frames)
        measured_eef_pose = read_dataset_or_empty(f, "measured_eef_pose", num_frames)
        command_eef_pose = read_dataset_or_empty(f, "command_eef_pose", num_frames)
        measured_eef_pose_rel = read_dataset_or_empty(f, "measured_eef_pose_rel", num_frames)
        command_eef_pose_rel = read_dataset_or_empty(f, "command_eef_pose_rel", num_frames)

    return EpisodeData(
        source=source,
        task_desc=task_desc,
        time_values=np.asarray(time_values, dtype=np.float64),
        measured_joint_pos=np.asarray(measured_joint_pos, dtype=np.float64),
        command_joint_pos=np.asarray(command_joint_pos, dtype=np.float64),
        measured_eef_pose=np.asarray(measured_eef_pose, dtype=np.float64),
        command_eef_pose=np.asarray(command_eef_pose, dtype=np.float64),
        measured_eef_pose_rel=np.asarray(measured_eef_pose_rel, dtype=np.float64),
        command_eef_pose_rel=np.asarray(command_eef_pose_rel, dtype=np.float64),
        video_streams=discover_episode_video_streams(source.episode_dir),
    )


def build_frame_indices(num_frames, args):
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.int64)

    frame_start = max(0, args.frame_start)
    frame_stop = num_frames if args.frame_stop is None else min(args.frame_stop, num_frames)
    if frame_stop <= frame_start:
        raise ValueError(f"Invalid frame range: start={frame_start}, stop={frame_stop}")
    if args.frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {args.frame_stride}")

    indices = np.arange(frame_start, frame_stop, args.frame_stride, dtype=np.int64)
    if args.max_frames_per_episode is not None and args.max_frames_per_episode > 0 and len(indices) > args.max_frames_per_episode:
        sample_positions = np.linspace(0, len(indices) - 1, args.max_frames_per_episode)
        indices = indices[np.unique(np.round(sample_positions).astype(np.int64))]
    return indices


def resolve_output_fps(time_values, frame_stride, requested_fps):
    if requested_fps is not None:
        if requested_fps <= 0:
            raise ValueError(f"fps must be positive, got {requested_fps}")
        return float(requested_fps)

    if len(time_values) >= 2:
        deltas = np.diff(time_values)
        deltas = deltas[deltas > 0]
        if len(deltas) > 0:
            native_fps = 1.0 / np.median(deltas)
            return max(native_fps / max(frame_stride, 1), 1.0)

    return 20.0


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


def pose_from_transform(transform):
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = transform[:3, 3]
    pose[3:] = quaternion_xyzw_from_rotation(transform[:3, :3])
    return pose


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


def integrate_stacked_eef_pose_rel(initial_pose, rel):
    initial_pose = np.asarray(initial_pose, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[1] == 0:
        return np.zeros((len(rel), 0), dtype=np.float64)
    if initial_pose.ndim == 2:
        initial_pose = initial_pose[0]
    if initial_pose.shape[0] % 7 != 0 or rel.shape[1] % 6 != 0:
        raise ValueError(f"Expected 7D initial pose blocks and 6D EEF deltas, got initial={initial_pose.shape}, rel={rel.shape}.")
    if initial_pose.shape[0] // 7 != rel.shape[1] // 6:
        raise ValueError(
            f"Initial pose EEF count and relative EEF count differ: {initial_pose.shape[0] // 7} vs {rel.shape[1] // 6}."
        )

    integrated = np.zeros((rel.shape[0], initial_pose.shape[0]), dtype=np.float64)
    if len(rel) == 0:
        return integrated

    integrated[0] = initial_pose
    for row in range(len(rel)):
        if row > 0:
            integrated[row] = integrated[row - 1]
        for pose_start, rel_start in zip(range(0, initial_pose.shape[0], 7), range(0, rel.shape[1], 6)):
            integrated[row, pose_start : pose_start + 7] = compose_pose_delta(
                integrated[row, pose_start : pose_start + 7],
                rel[row, rel_start : rel_start + 6],
            )
    return integrated


def build_urdf_zero_initial_pose(kinematics):
    zero_joints = np.zeros(max(kinematics.num_actuated_joints, ARM_DOF_FOR_FK), dtype=np.float64)
    pose = pose_from_transform(kinematics.forward_kinematics(zero_joints))
    return np.concatenate([pose.copy() for _ in ARM_LABELS])


def build_joint_initial_pose(joint_positions, kinematics):
    poses = []
    zero_pose = pose_from_transform(
        kinematics.forward_kinematics(np.zeros(max(kinematics.num_actuated_joints, ARM_DOF_FOR_FK), dtype=np.float64))
    )
    for arm_slice in ARM_SLICES:
        if has_slice(joint_positions, arm_slice):
            transform = kinematics.forward_kinematics(joint_positions[0, arm_slice][:ARM_DOF_FOR_FK])
            poses.append(pose_from_transform(transform))
        else:
            poses.append(zero_pose.copy())
    return np.concatenate(poses)


def source_absolute_pose(episode, source_name):
    return episode.measured_eef_pose if source_name == "measured" else episode.command_eef_pose


def source_joint_pos(episode, source_name):
    return episode.measured_joint_pos if source_name == "measured" else episode.command_joint_pos


def source_eef_rel(episode, source_name):
    return episode.measured_eef_pose_rel if source_name == "measured" else episode.command_eef_pose_rel


def resolve_episode_eef_mode(args, episode):
    if args.eef_mode != "auto":
        return args.eef_mode
    if episode.measured_eef_pose_rel.shape[1] >= 6 or episode.command_eef_pose_rel.shape[1] >= 6:
        return "relative"
    return "absolute"


def build_relative_initial_pose(episode, args, kinematics, source_name):
    if args.relative_origin == "absolute_first":
        absolute_pose = episode.measured_eef_pose if source_name == "command" else source_absolute_pose(episode, source_name)
        if absolute_pose.ndim == 2 and absolute_pose.shape[0] > 0 and absolute_pose.shape[1] >= 14:
            return absolute_pose[0]
        absolute_pose = source_absolute_pose(episode, source_name)
        if absolute_pose.ndim == 2 and absolute_pose.shape[0] > 0 and absolute_pose.shape[1] >= 14:
            return absolute_pose[0]

    if args.relative_origin == "joint_first":
        joint_pos = episode.measured_joint_pos if source_name == "command" else source_joint_pos(episode, source_name)
        if joint_pos.ndim == 2 and joint_pos.shape[0] > 0 and joint_pos.shape[1] > 0:
            return build_joint_initial_pose(joint_pos, kinematics)
        joint_pos = source_joint_pos(episode, source_name)
        if joint_pos.ndim == 2 and joint_pos.shape[0] > 0 and joint_pos.shape[1] > 0:
            return build_joint_initial_pose(joint_pos, kinematics)

    return build_urdf_zero_initial_pose(kinematics)


def build_relative_eef_pose_sequence(episode, args, kinematics, source_name):
    rel = source_eef_rel(episode, source_name)
    if rel.ndim != 2 or rel.shape[1] < 6:
        return np.zeros((len(episode.time_values), 0), dtype=np.float64)

    initial_pose = build_relative_initial_pose(episode, args, kinematics, source_name)
    return integrate_stacked_eef_pose_rel(initial_pose[: (rel.shape[1] // 6) * 7], rel)


def build_render_eef_poses(episode, args, kinematics):
    eef_mode = resolve_episode_eef_mode(args, episode)
    if eef_mode == "relative":
        return (
            build_relative_eef_pose_sequence(episode, args, kinematics, "measured"),
            build_relative_eef_pose_sequence(episode, args, kinematics, "command"),
            eef_mode,
        )
    return episode.measured_eef_pose, episode.command_eef_pose, eef_mode


def should_draw_urdf_reference(args, eef_mode):
    return eef_mode == "relative" and not args.hide_urdf_reference


def load_joint_limits(kinematics):
    root = ET.parse(kinematics.urdf_path).getroot()
    limits_by_name = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        limits_by_name[name] = (float(lower), float(upper))

    lower_limits = []
    upper_limits = []
    for joint in kinematics.chain:
        if joint["type"] == "fixed":
            continue
        lower, upper = limits_by_name.get(joint["name"], (-np.pi, np.pi))
        lower_limits.append(lower)
        upper_limits.append(upper)

    return np.asarray(lower_limits, dtype=np.float64), np.asarray(upper_limits, dtype=np.float64)


def weighted_pose_error(current_transform, target_pose, args):
    target_pose = np.asarray(target_pose, dtype=np.float64)
    current_rot = current_transform[:3, :3]
    target_rot = quaternion_to_rotation_matrix(target_pose[3:])
    position_error = target_pose[:3] - current_transform[:3, 3]
    rotation_error = rotation_vector_from_matrix(current_rot.T @ target_rot)
    weighted = np.concatenate(
        [
            position_error * args.ik_position_weight,
            rotation_error * args.ik_orientation_weight,
        ]
    )
    return weighted, position_error, rotation_error


def numerical_pose_jacobian(kinematics, joints, base_transform, args):
    dof = len(joints)
    jacobian = np.zeros((6, dof), dtype=np.float64)
    base_position = base_transform[:3, 3]
    base_rotation = base_transform[:3, :3]

    for joint_index in range(dof):
        perturbed = joints.copy()
        perturbed[joint_index] += args.ik_fd_eps
        perturbed_transform = kinematics.forward_kinematics(perturbed)
        position_delta = (perturbed_transform[:3, 3] - base_position) / args.ik_fd_eps
        rotation_delta = rotation_vector_from_matrix(base_rotation.T @ perturbed_transform[:3, :3]) / args.ik_fd_eps
        jacobian[:, joint_index] = np.concatenate(
            [
                position_delta * args.ik_position_weight,
                rotation_delta * args.ik_orientation_weight,
            ]
        )

    return jacobian


def geometric_pose_jacobian_from_trace(kinematics, joints, trace, current_transform, args):
    dof = len(joints)
    jacobian = np.zeros((6, dof), dtype=np.float64)
    eef_position = current_transform[:3, 3]
    eef_rotation = current_transform[:3, :3]
    joint_index = 0

    for entry in trace[1:]:
        joint_type = entry.get("type")
        if joint_type == "fixed":
            continue
        if joint_index >= dof:
            break

        pre_motion_transform = entry["pre_motion_transform"]
        axis = pre_motion_transform[:3, :3] @ np.asarray(entry["axis"], dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if not np.isclose(axis_norm, 0.0):
            axis = axis / axis_norm

        joint_origin = pre_motion_transform[:3, 3]
        if joint_type in ("revolute", "continuous"):
            linear = np.cross(axis, eef_position - joint_origin)
            angular = eef_rotation.T @ axis
        elif joint_type == "prismatic":
            linear = axis
            angular = np.zeros(3, dtype=np.float64)
        else:
            joint_index += 1
            continue

        jacobian[:, joint_index] = np.concatenate(
            [
                linear * args.ik_position_weight,
                angular * args.ik_orientation_weight,
            ]
        )
        joint_index += 1

    return jacobian


def solve_ik_pose(kinematics, target_pose, seed_joints, lower_limits, upper_limits, args):
    joints = np.clip(np.asarray(seed_joints, dtype=np.float64), lower_limits, upper_limits)
    damping_matrix = (args.ik_damping**2) * np.eye(6, dtype=np.float64)
    final_position_error = np.zeros(3, dtype=np.float64)
    final_rotation_error = np.zeros(3, dtype=np.float64)

    for iteration in range(max(args.ik_max_iterations, 1)):
        trace = kinematics.forward_kinematics_trace(joints)
        current_transform = trace[-1]["transform"]
        weighted_error, final_position_error, final_rotation_error = weighted_pose_error(current_transform, target_pose, args)
        if np.linalg.norm(weighted_error) <= args.ik_tolerance:
            return joints, final_position_error, final_rotation_error, iteration

        if args.ik_jacobian == "numerical":
            jacobian = numerical_pose_jacobian(kinematics, joints, current_transform, args)
        else:
            jacobian = geometric_pose_jacobian_from_trace(kinematics, joints, trace, current_transform, args)
        try:
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_matrix, weighted_error)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(jacobian) @ weighted_error

        step_norm = np.linalg.norm(step)
        if step_norm > args.ik_max_step:
            step *= args.ik_max_step / step_norm

        joints = np.clip(joints + step, lower_limits, upper_limits)

    current_transform = kinematics.forward_kinematics(joints)
    _, final_position_error, final_rotation_error = weighted_pose_error(current_transform, target_pose, args)
    return joints, final_position_error, final_rotation_error, max(args.ik_max_iterations, 1)


def selected_sources(args):
    if args.source == "both":
        return ("measured", "command")
    return (args.source,)


def selected_arm_indices(args):
    if args.arm == "both":
        return (0, 1)
    return (0,) if args.arm == "left" else (1,)


def initial_ik_seed(episode, source_name, arm_index, args, kinematics):
    dof = min(kinematics.num_actuated_joints, ARM_DOF_FOR_FK)
    if args.ik_seed == "joint_first":
        joint_positions = source_joint_pos(episode, source_name)
        arm_slice = ARM_SLICES[arm_index]
        if has_slice(joint_positions, arm_slice):
            return joint_positions[0, arm_slice][:dof]
    return np.zeros(dof, dtype=np.float64)


def solve_episode_ik(episode, args, kinematics, frame_indices, render_eef_poses=None):
    if not args.ik_preview:
        return None

    if args.ik_max_iterations <= 0:
        raise ValueError(f"ik_max_iterations must be positive, got {args.ik_max_iterations}")
    if args.ik_damping < 0.0:
        raise ValueError(f"ik_damping must be non-negative, got {args.ik_damping}")
    if args.ik_max_step <= 0.0:
        raise ValueError(f"ik_max_step must be positive, got {args.ik_max_step}")
    if args.ik_jacobian == "numerical" and args.ik_fd_eps <= 0.0:
        raise ValueError(f"ik_fd_eps must be positive, got {args.ik_fd_eps}")

    if render_eef_poses is None:
        render_eef_poses = build_render_eef_poses(episode, args, kinematics)
    measured_eef_pose, command_eef_pose, _ = render_eef_poses
    pose_by_source = {
        "measured": measured_eef_pose,
        "command": command_eef_pose,
    }
    lower_limits, upper_limits = load_joint_limits(kinematics)
    dof = min(kinematics.num_actuated_joints, ARM_DOF_FOR_FK)
    lower_limits = lower_limits[:dof]
    upper_limits = upper_limits[:dof]
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    frame_index_to_position = {int(frame_idx): index for index, frame_idx in enumerate(frame_indices)}

    joints_by_source = {}
    position_error_by_source = {}
    rotation_error_by_source = {}
    iterations_by_source = {}

    for source_name in selected_sources(args):
        poses = pose_by_source[source_name]
        joints = np.full((len(frame_indices), len(ARM_LABELS), dof), np.nan, dtype=np.float64)
        position_error = np.full((len(frame_indices), len(ARM_LABELS), 3), np.nan, dtype=np.float64)
        rotation_error = np.full((len(frame_indices), len(ARM_LABELS), 3), np.nan, dtype=np.float64)
        iterations = np.full((len(frame_indices), len(ARM_LABELS)), -1, dtype=np.int64)

        for arm_index in selected_arm_indices(args):
            eef_slice = EEF_SLICES[arm_index]
            if not has_slice(poses, eef_slice):
                continue

            seed = initial_ik_seed(episode, source_name, arm_index, args, kinematics)
            for local_index, frame_idx in enumerate(frame_indices):
                target_pose = poses[int(frame_idx), eef_slice]
                seed, pos_err, rot_err, iter_count = solve_ik_pose(
                    kinematics,
                    target_pose,
                    seed,
                    lower_limits,
                    upper_limits,
                    args,
                )
                joints[local_index, arm_index] = seed
                position_error[local_index, arm_index] = pos_err
                rotation_error[local_index, arm_index] = rot_err
                iterations[local_index, arm_index] = iter_count

        joints_by_source[source_name] = joints
        position_error_by_source[source_name] = position_error
        rotation_error_by_source[source_name] = rotation_error
        iterations_by_source[source_name] = iterations

    return EpisodeIkSolution(
        frame_indices=frame_indices,
        frame_index_to_position=frame_index_to_position,
        joints_by_source=joints_by_source,
        position_error_by_source=position_error_by_source,
        rotation_error_by_source=rotation_error_by_source,
        iterations_by_source=iterations_by_source,
    )


def get_ik_joints(ik_solution, source_name, arm_index, frame_idx):
    if ik_solution is None or source_name not in ik_solution.joints_by_source:
        return None
    local_index = ik_solution.frame_index_to_position.get(int(frame_idx))
    if local_index is None:
        return None
    joints = ik_solution.joints_by_source[source_name][local_index, arm_index]
    if not np.all(np.isfinite(joints)):
        return None
    return joints


def print_ik_summary(episode_name, ik_solution):
    if ik_solution is None:
        return

    for source_name in sorted(ik_solution.joints_by_source):
        pos_norm = np.linalg.norm(ik_solution.position_error_by_source[source_name], axis=2)
        rot_norm = np.linalg.norm(ik_solution.rotation_error_by_source[source_name], axis=2)
        valid = np.isfinite(pos_norm)
        if not np.any(valid):
            print(f"IK {episode_name} {source_name}: no valid targets")
            continue
        print(
            f"IK {episode_name} {source_name}: "
            f"pos_err median/max={np.nanmedian(pos_norm):.5f}/{np.nanmax(pos_norm):.5f} m, "
            f"rot_err median/max={np.nanmedian(rot_norm):.5f}/{np.nanmax(rot_norm):.5f} rad"
        )


def plot_eef_axes(ax, pose, offset, axis_length, alpha=1.0):
    origin = pose[:3] + offset
    rotation = quaternion_to_rotation_matrix(pose[3:])
    for axis_index, axis_color in enumerate(EEF_AXIS_COLORS):
        tip = origin + rotation[:, axis_index] * axis_length
        ax.plot(
            [origin[0], tip[0]],
            [origin[1], tip[1]],
            [origin[2], tip[2]],
            color=axis_color,
            linewidth=2.0,
            alpha=alpha,
        )


def plot_arm_skeleton(ax, kinematics, joints, color, offset, linestyle, alpha, linewidth):
    trace = kinematics.forward_kinematics_trace(joints[:ARM_DOF_FOR_FK])
    points = np.array([entry["transform"][:3, 3] + offset for entry in trace], dtype=np.float64)
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=18, alpha=alpha)
    return points


def plot_eef_trajectory(ax, poses, frame_idx, color, offset, linestyle, alpha_full, alpha_past, label):
    positions = poses[:, :3] + offset
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color=color,
        linestyle=linestyle,
        linewidth=1.0,
        alpha=alpha_full,
    )
    past = positions[: frame_idx + 1]
    ax.plot(
        past[:, 0],
        past[:, 1],
        past[:, 2],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        alpha=alpha_past,
        label=label,
    )
    current = positions[frame_idx]
    ax.scatter([current[0]], [current[1]], [current[2]], color=color, s=45, alpha=0.95)


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).max() * 0.55, 0.18)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def gather_points_for_bounds(episode, frame_indices, args, kinematics, ik_solution=None, render_eef_poses=None):
    if len(frame_indices) == 0:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    if args.bounds_samples is not None and args.bounds_samples > 0 and len(frame_indices) > args.bounds_samples:
        sample_positions = np.linspace(0, len(frame_indices) - 1, args.bounds_samples)
        frame_indices = frame_indices[np.unique(np.round(sample_positions).astype(np.int64))]

    if render_eef_poses is None:
        render_eef_poses = build_render_eef_poses(episode, args, kinematics)
    measured_eef_pose, command_eef_pose, eef_mode = render_eef_poses
    points = []
    if should_draw_urdf_reference(args, eef_mode):
        zero_joints = np.zeros(max(kinematics.num_actuated_joints, ARM_DOF_FOR_FK), dtype=np.float64)
        for arm_name, _, offset in ARM_LABELS:
            if args.arm != "both" and args.arm != arm_name:
                continue
            trace = kinematics.forward_kinematics_trace(zero_joints)
            points.extend(entry["transform"][:3, 3] + offset for entry in trace)

    for frame_idx in frame_indices:
        for arm_index, (arm_name, _, offset) in enumerate(ARM_LABELS):
            if args.arm != "both" and args.arm != arm_name:
                continue

            arm_slice = ARM_SLICES[arm_index]
            eef_slice = EEF_SLICES[arm_index]

            if args.source in ("measured", "both"):
                if has_slice(episode.measured_joint_pos, arm_slice):
                    trace = kinematics.forward_kinematics_trace(episode.measured_joint_pos[frame_idx, arm_slice][:ARM_DOF_FOR_FK])
                    points.extend(entry["transform"][:3, 3] + offset for entry in trace)
                if has_slice(measured_eef_pose, eef_slice):
                    points.append(measured_eef_pose[frame_idx, eef_slice][:3] + offset)
                ik_joints = get_ik_joints(ik_solution, "measured", arm_index, frame_idx)
                if ik_joints is not None:
                    trace = kinematics.forward_kinematics_trace(ik_joints)
                    points.extend(entry["transform"][:3, 3] + offset for entry in trace)

            if args.source in ("command", "both"):
                if has_slice(episode.command_joint_pos, arm_slice):
                    trace = kinematics.forward_kinematics_trace(episode.command_joint_pos[frame_idx, arm_slice][:ARM_DOF_FOR_FK])
                    points.extend(entry["transform"][:3, 3] + offset for entry in trace)
                if has_slice(command_eef_pose, eef_slice):
                    points.append(command_eef_pose[frame_idx, eef_slice][:3] + offset)
                ik_joints = get_ik_joints(ik_solution, "command", arm_index, frame_idx)
                if ik_joints is not None:
                    trace = kinematics.forward_kinematics_trace(ik_joints)
                    points.extend(entry["transform"][:3, 3] + offset for entry in trace)

    if not points:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return np.array(points, dtype=np.float64)


def format_time_axis(ax, time_values):
    ax.set_xlim(time_values[0], time_values[-1] if len(time_values) > 1 else time_values[0] + 1.0)
    ax.grid(True, alpha=0.25)


def plot_joint_panel(ax, episode, frame_idx, arm_index, arm_name, args, ik_solution=None):
    arm_slice = ARM_SLICES[arm_index]
    time_values = episode.time_values
    current_time = time_values[frame_idx]
    plotted = False

    if args.source in ("measured", "both") and has_slice(episode.measured_joint_pos, arm_slice):
        for joint_offset in range(arm_slice.stop - arm_slice.start):
            series = episode.measured_joint_pos[:, arm_slice.start + joint_offset]
            ax.plot(time_values, series, color=JOINT_COLORS[joint_offset], linewidth=1.2, alpha=0.85)
            ax.scatter([current_time], [series[frame_idx]], color=JOINT_COLORS[joint_offset], s=12, alpha=0.95)
        plotted = True

    if args.source in ("command", "both") and has_slice(episode.command_joint_pos, arm_slice):
        for joint_offset in range(arm_slice.stop - arm_slice.start):
            series = episode.command_joint_pos[:, arm_slice.start + joint_offset]
            ax.plot(time_values, series, color=JOINT_COLORS[joint_offset], linewidth=1.0, linestyle="--", alpha=0.55)
            ax.scatter([current_time], [series[frame_idx]], color=JOINT_COLORS[joint_offset], s=10, alpha=0.65)
        plotted = True

    if ik_solution is not None:
        local_index = ik_solution.frame_index_to_position.get(int(frame_idx))
        ik_time_values = episode.time_values[ik_solution.frame_indices]
        if local_index is not None and args.source in ("measured", "both") and "measured" in ik_solution.joints_by_source:
            ik_joints = ik_solution.joints_by_source["measured"][:, arm_index, :]
            if np.any(np.isfinite(ik_joints)):
                for joint_offset in range(ik_joints.shape[1]):
                    series = ik_joints[:, joint_offset]
                    ax.plot(ik_time_values, series, color=JOINT_COLORS[joint_offset], linewidth=1.4, alpha=0.90)
                    ax.scatter([current_time], [series[local_index]], color=JOINT_COLORS[joint_offset], s=14, alpha=0.95)
                plotted = True
        if local_index is not None and args.source in ("command", "both") and "command" in ik_solution.joints_by_source:
            ik_joints = ik_solution.joints_by_source["command"][:, arm_index, :]
            if np.any(np.isfinite(ik_joints)):
                for joint_offset in range(ik_joints.shape[1]):
                    series = ik_joints[:, joint_offset]
                    ax.plot(ik_time_values, series, color=JOINT_COLORS[joint_offset], linewidth=1.1, linestyle="--", alpha=0.65)
                    ax.scatter([current_time], [series[local_index]], color=JOINT_COLORS[joint_offset], s=12, alpha=0.70)
                plotted = True

    if not plotted:
        disable_axis(ax, f"{arm_name.capitalize()} joints")
        return

    ax.axvline(current_time, color="#111111", linewidth=1.0, alpha=0.6)
    format_time_axis(ax, time_values)
    ax.set_ylabel("joint [rad]")
    suffix = " (IK preview)" if ik_solution is not None else ""
    ax.set_title(f"{arm_name.capitalize()} joints{suffix}")


def plot_eef_panel(ax, episode, frame_idx, arm_index, arm_name, args):
    eef_mode = resolve_episode_eef_mode(args, episode)
    if eef_mode == "relative":
        eef_slice = EEF_REL_SLICES[arm_index]
        measured_source = episode.measured_eef_pose_rel
        command_source = episode.command_eef_pose_rel
        ylabel = "eef rel dxyz [m]"
        title = f"{arm_name.capitalize()} EEF rel dxyz"
    else:
        eef_slice = EEF_SLICES[arm_index]
        measured_source = episode.measured_eef_pose
        command_source = episode.command_eef_pose
        ylabel = "eef xyz [m]"
        title = f"{arm_name.capitalize()} EEF xyz"

    time_values = episode.time_values
    current_time = time_values[frame_idx]
    plotted = False

    if args.source in ("measured", "both") and has_slice(measured_source, eef_slice):
        measured = measured_source[:, eef_slice][:, :3]
        for axis_index, axis_color in enumerate(EEF_AXIS_COLORS):
            series = measured[:, axis_index]
            ax.plot(time_values, series, color=axis_color, linewidth=1.3, alpha=0.85)
            ax.scatter([current_time], [series[frame_idx]], color=axis_color, s=14, alpha=0.95)
        plotted = True

    if args.source in ("command", "both") and has_slice(command_source, eef_slice):
        command = command_source[:, eef_slice][:, :3]
        for axis_index, axis_color in enumerate(EEF_AXIS_COLORS):
            series = command[:, axis_index]
            ax.plot(time_values, series, color=axis_color, linewidth=1.0, linestyle="--", alpha=0.55)
            ax.scatter([current_time], [series[frame_idx]], color=axis_color, s=12, alpha=0.65)
        plotted = True

    if not plotted:
        disable_axis(ax, f"{arm_name.capitalize()} EEF xyz")
        return

    ax.axvline(current_time, color="#111111", linewidth=1.0, alpha=0.6)
    format_time_axis(ax, time_values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def disable_axis(ax, title):
    ax.axis("off")
    ax.set_title(title)


def open_episode_video_captures(episode):
    if getattr(episode, "video_streams", None) is None:
        return []
    cv2 = load_cv2()
    captures = []
    for stream in episode.video_streams:
        capture = cv2.VideoCapture(str(stream.path))
        if not capture.isOpened():
            print(f"⚠️  Failed to open episode video: {stream.path}")
            continue
        captures.append(EpisodeVideoCapture(stream=stream, capture=capture))
    return captures


def close_episode_video_captures(captures):
    for video_capture in captures:
        video_capture.capture.release()


def read_video_frame_rgb(video_capture, frame_idx):
    cv2 = load_cv2()
    capture = video_capture.capture
    frame_idx = max(int(frame_idx), 0)

    if frame_idx < video_capture.next_frame_idx:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        video_capture.next_frame_idx = frame_idx

    frame_bgr = None
    while video_capture.next_frame_idx <= frame_idx:
        ok, current_bgr = capture.read()
        if not ok or current_bgr is None:
            return None
        frame_bgr = current_bgr
        video_capture.next_frame_idx += 1

    if frame_bgr is None:
        return None
    return frame_bgr[:, :, ::-1]


def build_camera_panel(captures, frame_idx):
    if not captures:
        return None, None

    cv2 = load_cv2()
    tiles = []
    labels = []
    if len(captures) == 1:
        target_height = 360
        grid_cols = 1
    elif len(captures) == 2:
        target_height = 300
        grid_cols = 2
    else:
        target_height = 240
        grid_cols = 2

    for video_capture in captures:
        frame_rgb = read_video_frame_rgb(video_capture, frame_idx)
        if frame_rgb is None:
            continue
        scale = target_height / max(frame_rgb.shape[0], 1)
        target_width = max(1, int(round(frame_rgb.shape[1] * scale)))
        tile_bgr = cv2.resize(frame_rgb[:, :, ::-1], (target_width, target_height), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile_bgr,
            video_capture.stream.name,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile_bgr,
            video_capture.stream.name,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile_bgr[:, :, ::-1])
        labels.append(video_capture.stream.name)

    if not tiles:
        return None, None

    if len(tiles) == 1:
        return tiles[0], labels[0]

    pad = 8
    grid_rows = int(np.ceil(len(tiles) / grid_cols))
    max_width = max(tile.shape[1] for tile in tiles)
    max_height = max(tile.shape[0] for tile in tiles)
    blank_tile = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    normalized_tiles = []
    for tile in tiles:
        pad_bottom = max_height - tile.shape[0]
        pad_right = max_width - tile.shape[1]
        normalized_tiles.append(np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="constant"))

    while len(normalized_tiles) < grid_rows * grid_cols:
        normalized_tiles.append(blank_tile.copy())

    row_images = []
    horizontal_spacer = np.full((max_height, pad, 3), 12, dtype=np.uint8)
    vertical_spacer = np.full((pad, grid_cols * max_width + (grid_cols - 1) * pad, 3), 12, dtype=np.uint8)
    for row_index in range(grid_rows):
        row_tiles = normalized_tiles[row_index * grid_cols : (row_index + 1) * grid_cols]
        row_image = row_tiles[0]
        for tile in row_tiles[1:]:
            row_image = np.concatenate([row_image, horizontal_spacer, tile], axis=1)
        row_images.append(row_image)

    mosaic = row_images[0]
    for row_image in row_images[1:]:
        mosaic = np.concatenate([mosaic, vertical_spacer, row_image], axis=0)
    return mosaic, " | ".join(labels)


def render_camera_panel(ax, camera_panel_rgb, camera_panel_title):
    if camera_panel_rgb is None:
        disable_axis(ax, "Episode video")
        return
    ax.imshow(camera_panel_rgb)
    ax.axis("off")
    ax.set_title(f"Episode video | {camera_panel_title}", fontsize=12)


def render_episode_frame(
    plt,
    episode,
    frame_idx,
    frame_position,
    total_frames,
    args,
    kinematics,
    axes_points,
    camera_panel_rgb=None,
    camera_panel_title=None,
    ik_solution=None,
    render_eef_poses=None,
):
    if render_eef_poses is None:
        render_eef_poses = build_render_eef_poses(episode, args, kinematics)
    measured_eef_pose, command_eef_pose, eef_mode = render_eef_poses
    figure_scale = max(float(args.figure_scale), 0.2)
    fig = plt.figure(figsize=(22 * figure_scale, 12.8 * figure_scale), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.08)
    ax_3d = fig.add_subplot(outer[0, 0], projection="3d")
    right = outer[0, 1].subgridspec(3, 2, height_ratios=[1.35, 1.0, 1.0], hspace=0.30, wspace=0.18)
    ax_video = fig.add_subplot(right[0, :])
    ax_joint_left = fig.add_subplot(right[1, 0])
    ax_joint_right = fig.add_subplot(right[1, 1])
    ax_eef_left = fig.add_subplot(right[2, 0])
    ax_eef_right = fig.add_subplot(right[2, 1])

    current_time = episode.time_values[frame_idx]
    fig.suptitle(
        f"RMB preview | {episode.source.dataset_name} | {episode.source.episode_name} | "
        f"frame {frame_position + 1}/{total_frames} | t={current_time:.3f}s\n"
        f"task: {episode.task_desc}",
        fontsize=16,
    )

    render_camera_panel(ax_video, camera_panel_rgb, camera_panel_title)

    for arm_index, (arm_name, arm_color, offset) in enumerate(ARM_LABELS):
        if args.arm != "both" and args.arm != arm_name:
            continue

        arm_slice = ARM_SLICES[arm_index]
        eef_slice = EEF_SLICES[arm_index]

        if should_draw_urdf_reference(args, eef_mode):
            plot_arm_skeleton(
                ax_3d,
                kinematics,
                np.zeros(max(kinematics.num_actuated_joints, ARM_DOF_FOR_FK), dtype=np.float64),
                color=arm_color,
                offset=offset,
                linestyle=":",
                alpha=0.22,
                linewidth=1.3,
            )

        if args.source in ("measured", "both"):
            if has_slice(episode.measured_joint_pos, arm_slice):
                plot_arm_skeleton(
                    ax_3d,
                    kinematics,
                    episode.measured_joint_pos[frame_idx, arm_slice],
                    color=arm_color,
                    offset=offset,
                    linestyle="-",
                    alpha=0.95,
                    linewidth=2.6,
                )
            if has_slice(measured_eef_pose, eef_slice):
                plot_eef_trajectory(
                    ax_3d,
                    measured_eef_pose[:, eef_slice],
                    frame_idx=frame_idx,
                    color=arm_color,
                    offset=offset,
                    linestyle="-",
                    alpha_full=0.10,
                    alpha_past=0.95,
                    label=f"{arm_name} measured {eef_mode}",
                )
                plot_eef_axes(
                    ax_3d,
                    measured_eef_pose[frame_idx, eef_slice],
                    offset=offset,
                    axis_length=args.eef_axis_length,
                    alpha=0.95,
                )
            ik_joints = get_ik_joints(ik_solution, "measured", arm_index, frame_idx)
            if ik_joints is not None:
                plot_arm_skeleton(
                    ax_3d,
                    kinematics,
                    ik_joints,
                    color=arm_color,
                    offset=offset,
                    linestyle="-.",
                    alpha=0.85,
                    linewidth=2.2,
                )

        if args.source in ("command", "both"):
            if has_slice(episode.command_joint_pos, arm_slice):
                plot_arm_skeleton(
                    ax_3d,
                    kinematics,
                    episode.command_joint_pos[frame_idx, arm_slice],
                    color=arm_color,
                    offset=offset,
                    linestyle="--",
                    alpha=0.65,
                    linewidth=1.8,
                )
            if has_slice(command_eef_pose, eef_slice):
                plot_eef_trajectory(
                    ax_3d,
                    command_eef_pose[:, eef_slice],
                    frame_idx=frame_idx,
                    color=arm_color,
                    offset=offset,
                    linestyle="--",
                    alpha_full=0.08,
                    alpha_past=0.60,
                    label=f"{arm_name} command {eef_mode}",
                )
                plot_eef_axes(
                    ax_3d,
                    command_eef_pose[frame_idx, eef_slice],
                    offset=offset,
                    axis_length=args.eef_axis_length * 0.85,
                    alpha=0.65,
                )
            ik_joints = get_ik_joints(ik_solution, "command", arm_index, frame_idx)
            if ik_joints is not None:
                plot_arm_skeleton(
                    ax_3d,
                    kinematics,
                    ik_joints,
                    color=arm_color,
                    offset=offset,
                    linestyle=":",
                    alpha=0.72,
                    linewidth=2.0,
                )

    set_axes_equal(ax_3d, axes_points)
    ik_suffix = " + IK" if ik_solution is not None else ""
    ax_3d.set_title(f"Arm pose + EEF trajectory ({eef_mode}{ik_suffix})")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.grid(True, alpha=0.25)
    if args.source == "both":
        ax_3d.legend(loc="upper right")

    if args.arm in ("left", "both"):
        plot_joint_panel(ax_joint_left, episode, frame_idx, 0, "left", args, ik_solution=ik_solution)
        plot_eef_panel(ax_eef_left, episode, frame_idx, 0, "left", args)
        ax_eef_left.set_xlabel("time [s]")
    else:
        disable_axis(ax_joint_left, "Left joints")
        disable_axis(ax_eef_left, "Left EEF xyz")

    if args.arm in ("right", "both"):
        plot_joint_panel(ax_joint_right, episode, frame_idx, 1, "right", args, ik_solution=ik_solution)
        plot_eef_panel(ax_eef_right, episode, frame_idx, 1, "right", args)
        ax_eef_right.set_xlabel("time [s]")
    else:
        disable_axis(ax_joint_right, "Right joints")
        disable_axis(ax_eef_right, "Right EEF xyz")

    return fig


def figure_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[:, :, :3]
    return rgb[:, :, ::-1].copy()


def open_video_writer(output_path, probe_bgr, fps):
    cv2 = load_cv2()
    height, width = probe_bgr.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {output_path}")
    return writer


def render_dataset_preview(args, plt, robot_urdf, episode_sources, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kinematics = UrdfKinematics(robot_urdf)

    first_episode = load_episode_data(episode_sources[0])
    first_indices = build_frame_indices(len(first_episode.time_values), args)
    if len(first_indices) == 0:
        raise ValueError(f"No renderable frames in {first_episode.source.hdf5_path}")
    output_fps = resolve_output_fps(first_episode.time_values, args.frame_stride, args.fps)
    first_render_eef_poses = build_render_eef_poses(first_episode, args, kinematics)
    first_ik_solution = solve_episode_ik(first_episode, args, kinematics, first_indices, render_eef_poses=first_render_eef_poses)
    first_axes_points = gather_points_for_bounds(
        first_episode,
        first_indices,
        args,
        kinematics,
        ik_solution=first_ik_solution,
        render_eef_poses=first_render_eef_poses,
    )
    first_captures = [] if args.no_camera else open_episode_video_captures(first_episode)
    first_camera_panel_rgb, first_camera_panel_title = (
        (None, None) if args.no_camera else build_camera_panel(first_captures, int(first_indices[0]))
    )
    probe_fig = render_episode_frame(
        plt,
        first_episode,
        frame_idx=int(first_indices[0]),
        frame_position=0,
        total_frames=len(first_indices),
        args=args,
        kinematics=kinematics,
        axes_points=first_axes_points,
        camera_panel_rgb=first_camera_panel_rgb,
        camera_panel_title=first_camera_panel_title,
        ik_solution=first_ik_solution,
        render_eef_poses=first_render_eef_poses,
    )
    probe_bgr = figure_to_bgr(probe_fig)
    plt.close(probe_fig)
    close_episode_video_captures(first_captures)

    writer = open_video_writer(output_path, probe_bgr, output_fps)
    print(f"output: {output_path}")
    print(f"robot_urdf: {robot_urdf}")
    print(f"eef_mode: {first_render_eef_poses[2]}")
    if first_render_eef_poses[2] == "relative":
        print(f"relative_origin: {args.relative_origin}")
    if args.ik_preview:
        print(
            "ik_preview: enabled "
            f"(seed={args.ik_seed}, jacobian={args.ik_jacobian}, "
            f"max_iterations={args.ik_max_iterations}, damping={args.ik_damping})"
        )
        print_ik_summary(first_episode.source.episode_name, first_ik_solution)
    print(f"episodes: {len(episode_sources)}")
    print(f"video_fps: {output_fps:.3f}")

    try:
        for episode_index, source in enumerate(episode_sources):
            episode = first_episode if episode_index == 0 else load_episode_data(source)
            frame_indices = build_frame_indices(len(episode.time_values), args)
            if len(frame_indices) == 0:
                print(f"⚠️  Skipping empty frame selection: {source.hdf5_path}")
                continue

            render_eef_poses = first_render_eef_poses if episode_index == 0 else build_render_eef_poses(episode, args, kinematics)
            ik_solution = (
                first_ik_solution
                if episode_index == 0
                else solve_episode_ik(episode, args, kinematics, frame_indices, render_eef_poses=render_eef_poses)
            )
            if args.ik_preview and episode_index != 0:
                print_ik_summary(source.episode_name, ik_solution)

            axes_points = (
                first_axes_points
                if episode_index == 0
                else gather_points_for_bounds(
                    episode,
                    frame_indices,
                    args,
                    kinematics,
                    ik_solution=ik_solution,
                    render_eef_poses=render_eef_poses,
                )
            )
            if len(frame_indices) > 500 and args.max_frames_per_episode is None:
                print(
                    f"⚠️  {source.episode_name} will render {len(frame_indices)} frames. "
                    "Use --max_frames_per_episode to make previews faster."
                )

            print(
                f"🎬 Episode {episode_index + 1}/{len(episode_sources)}: {source.episode_name} "
                f"(frames={len(frame_indices)}, original={len(episode.time_values)})"
            )

            captures = [] if args.no_camera else open_episode_video_captures(episode)
            try:
                for local_index, frame_idx in enumerate(frame_indices):
                    camera_panel_rgb, camera_panel_title = (
                        (None, None) if args.no_camera else build_camera_panel(captures, int(frame_idx))
                    )
                    fig = render_episode_frame(
                        plt,
                        episode,
                        frame_idx=int(frame_idx),
                        frame_position=local_index,
                        total_frames=len(frame_indices),
                        args=args,
                        kinematics=kinematics,
                        axes_points=axes_points,
                        camera_panel_rgb=camera_panel_rgb,
                        camera_panel_title=camera_panel_title,
                        ik_solution=ik_solution,
                        render_eef_poses=render_eef_poses,
                    )
                    writer.write(figure_to_bgr(fig))
                    plt.close(fig)
            finally:
                close_episode_video_captures(captures)
    finally:
        writer.release()

    print(f"saved: {output_path}")


def main():
    args = parse_args()
    episode_sources = resolve_multiple_episode_sources(args.input_dir, args.max_episodes)
    output_path = resolve_output_path(args.output_dir, args.output_path, args.input_dir, episode_sources)
    robot_urdf = resolve_robot_urdf(args.robot_urdf)
    plt = load_matplotlib(args.backend)
    render_dataset_preview(args, plt, robot_urdf, episode_sources, output_path)


if __name__ == "__main__":
    main()
