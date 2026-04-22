#!/usr/bin/env python3

import argparse
import os
import sys
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
    quaternion_to_rotation_matrix,
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
JOINT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")
EEF_AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")


@dataclass(frozen=True)
class EpisodeSource:
    dataset_name: str
    episode_name: str
    episode_dir: Path
    hdf5_path: Path


@dataclass
class EpisodeData:
    source: EpisodeSource
    task_desc: str
    time_values: np.ndarray
    measured_joint_pos: np.ndarray
    command_joint_pos: np.ndarray
    measured_eef_pose: np.ndarray
    command_eef_pose: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Render an RMB episode or dataset preview into a single MP4.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to main.rmb.hdf5, an episode_*.rmb directory, or a dataset directory containing episode_*.rmb.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output MP4 path. Defaults next to the input episode/dataset.",
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
        help="Maximum number of episodes to include when input_path is a dataset directory.",
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


def resolve_episode_sources(input_path, max_episodes):
    input_path = Path(input_path).expanduser().resolve()

    if input_path.is_file():
        if input_path.name != "main.rmb.hdf5":
            raise FileNotFoundError(f"Expected an RMB file named main.rmb.hdf5, got: {input_path}")
        episode_dir = input_path.parent
        return [
            EpisodeSource(
                dataset_name=episode_dir.parent.name,
                episode_name=episode_dir.name,
                episode_dir=episode_dir,
                hdf5_path=input_path,
            )
        ]

    if input_path.is_dir() and (input_path / "main.rmb.hdf5").is_file():
        return [
            EpisodeSource(
                dataset_name=input_path.parent.name,
                episode_name=input_path.name,
                episode_dir=input_path,
                hdf5_path=(input_path / "main.rmb.hdf5"),
            )
        ]

    if input_path.is_dir():
        episode_dirs = sorted(path for path in input_path.iterdir() if path.is_dir() and path.name.endswith(".rmb"))
        if episode_dirs:
            if max_episodes is not None and max_episodes > 0:
                episode_dirs = episode_dirs[:max_episodes]
            sources = [
                EpisodeSource(
                    dataset_name=input_path.name,
                    episode_name=episode_dir.name,
                    episode_dir=episode_dir,
                    hdf5_path=(episode_dir / "main.rmb.hdf5"),
                )
                for episode_dir in episode_dirs
                if (episode_dir / "main.rmb.hdf5").is_file()
            ]
            if sources:
                return sources

    raise FileNotFoundError(
        "input_path must be main.rmb.hdf5, an episode_*.rmb directory, or a dataset directory containing episode_*.rmb"
    )


def build_default_output_path(input_path, episode_sources):
    input_path = Path(input_path).expanduser().resolve()
    if len(episode_sources) == 1:
        return episode_sources[0].episode_dir / "rmb_preview.mp4"
    return input_path / "rmb_dataset_preview.mp4"


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


def load_episode_data(source):
    with h5py.File(source.hdf5_path, "r") as f:
        task_desc = decode_attr(f.attrs.get("task_desc", source.dataset_name.replace("_", " ")))
        time_values = f["time"][()] if "time" in f else np.arange(len(f["measured_joint_pos"]), dtype=np.float64)
        measured_joint_pos = f["measured_joint_pos"][()]
        command_joint_pos = f["command_joint_pos"][()]
        measured_eef_pose = f["measured_eef_pose"][()]
        command_eef_pose = f["command_eef_pose"][()]

    return EpisodeData(
        source=source,
        task_desc=task_desc,
        time_values=np.asarray(time_values, dtype=np.float64),
        measured_joint_pos=np.asarray(measured_joint_pos, dtype=np.float64),
        command_joint_pos=np.asarray(command_joint_pos, dtype=np.float64),
        measured_eef_pose=np.asarray(measured_eef_pose, dtype=np.float64),
        command_eef_pose=np.asarray(command_eef_pose, dtype=np.float64),
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


def gather_points_for_bounds(episode, frame_indices, args, kinematics):
    if len(frame_indices) == 0:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    if args.bounds_samples is not None and args.bounds_samples > 0 and len(frame_indices) > args.bounds_samples:
        sample_positions = np.linspace(0, len(frame_indices) - 1, args.bounds_samples)
        frame_indices = frame_indices[np.unique(np.round(sample_positions).astype(np.int64))]

    points = []
    for frame_idx in frame_indices:
        for arm_index, (arm_name, _, offset) in enumerate(ARM_LABELS):
            if args.arm != "both" and args.arm != arm_name:
                continue

            arm_slice = ARM_SLICES[arm_index]
            eef_slice = EEF_SLICES[arm_index]

            if args.source in ("measured", "both"):
                trace = kinematics.forward_kinematics_trace(episode.measured_joint_pos[frame_idx, arm_slice][:ARM_DOF_FOR_FK])
                points.extend(entry["transform"][:3, 3] + offset for entry in trace)
                points.append(episode.measured_eef_pose[frame_idx, eef_slice][:3] + offset)

            if args.source in ("command", "both"):
                trace = kinematics.forward_kinematics_trace(episode.command_joint_pos[frame_idx, arm_slice][:ARM_DOF_FOR_FK])
                points.extend(entry["transform"][:3, 3] + offset for entry in trace)
                points.append(episode.command_eef_pose[frame_idx, eef_slice][:3] + offset)

    if not points:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return np.array(points, dtype=np.float64)


def format_time_axis(ax, time_values):
    ax.set_xlim(time_values[0], time_values[-1] if len(time_values) > 1 else time_values[0] + 1.0)
    ax.grid(True, alpha=0.25)


def plot_joint_panel(ax, episode, frame_idx, arm_index, arm_name, args):
    arm_slice = ARM_SLICES[arm_index]
    time_values = episode.time_values
    current_time = time_values[frame_idx]

    if args.source in ("measured", "both"):
        for joint_offset in range(arm_slice.stop - arm_slice.start):
            series = episode.measured_joint_pos[:, arm_slice.start + joint_offset]
            ax.plot(time_values, series, color=JOINT_COLORS[joint_offset], linewidth=1.2, alpha=0.85)
            ax.scatter([current_time], [series[frame_idx]], color=JOINT_COLORS[joint_offset], s=12, alpha=0.95)

    if args.source in ("command", "both"):
        for joint_offset in range(arm_slice.stop - arm_slice.start):
            series = episode.command_joint_pos[:, arm_slice.start + joint_offset]
            ax.plot(time_values, series, color=JOINT_COLORS[joint_offset], linewidth=1.0, linestyle="--", alpha=0.55)
            ax.scatter([current_time], [series[frame_idx]], color=JOINT_COLORS[joint_offset], s=10, alpha=0.65)

    ax.axvline(current_time, color="#111111", linewidth=1.0, alpha=0.6)
    format_time_axis(ax, time_values)
    ax.set_ylabel("joint [rad]")
    ax.set_title(f"{arm_name.capitalize()} joints")


def plot_eef_panel(ax, episode, frame_idx, arm_index, arm_name, args):
    eef_slice = EEF_SLICES[arm_index]
    time_values = episode.time_values
    current_time = time_values[frame_idx]

    if args.source in ("measured", "both"):
        measured = episode.measured_eef_pose[:, eef_slice][:, :3]
        for axis_index, axis_color in enumerate(EEF_AXIS_COLORS):
            series = measured[:, axis_index]
            ax.plot(time_values, series, color=axis_color, linewidth=1.3, alpha=0.85)
            ax.scatter([current_time], [series[frame_idx]], color=axis_color, s=14, alpha=0.95)

    if args.source in ("command", "both"):
        command = episode.command_eef_pose[:, eef_slice][:, :3]
        for axis_index, axis_color in enumerate(EEF_AXIS_COLORS):
            series = command[:, axis_index]
            ax.plot(time_values, series, color=axis_color, linewidth=1.0, linestyle="--", alpha=0.55)
            ax.scatter([current_time], [series[frame_idx]], color=axis_color, s=12, alpha=0.65)

    ax.axvline(current_time, color="#111111", linewidth=1.0, alpha=0.6)
    format_time_axis(ax, time_values)
    ax.set_ylabel("eef xyz [m]")
    ax.set_title(f"{arm_name.capitalize()} EEF xyz")


def disable_axis(ax, title):
    ax.axis("off")
    ax.set_title(title)


def render_episode_frame(plt, episode, frame_idx, frame_position, total_frames, args, kinematics, axes_points):
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0], hspace=0.36, wspace=0.24)
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_joint_left = fig.add_subplot(grid[0, 1])
    ax_joint_right = fig.add_subplot(grid[1, 1])
    ax_eef_left = fig.add_subplot(grid[2, 1])
    ax_eef_right = fig.add_subplot(grid[3, 1])

    current_time = episode.time_values[frame_idx]
    fig.suptitle(
        f"RMB preview | {episode.source.dataset_name} | {episode.source.episode_name} | "
        f"frame {frame_position + 1}/{total_frames} | t={current_time:.3f}s\n"
        f"task: {episode.task_desc}",
        fontsize=13,
    )

    for arm_index, (arm_name, arm_color, offset) in enumerate(ARM_LABELS):
        if args.arm != "both" and args.arm != arm_name:
            continue

        arm_slice = ARM_SLICES[arm_index]
        eef_slice = EEF_SLICES[arm_index]

        if args.source in ("measured", "both"):
            measured_points = plot_arm_skeleton(
                ax_3d,
                kinematics,
                episode.measured_joint_pos[frame_idx, arm_slice],
                color=arm_color,
                offset=offset,
                linestyle="-",
                alpha=0.95,
                linewidth=2.6,
            )
            plot_eef_trajectory(
                ax_3d,
                episode.measured_eef_pose[:, eef_slice],
                frame_idx=frame_idx,
                color=arm_color,
                offset=offset,
                linestyle="-",
                alpha_full=0.10,
                alpha_past=0.95,
                label=f"{arm_name} measured",
            )
            plot_eef_axes(
                ax_3d,
                episode.measured_eef_pose[frame_idx, eef_slice],
                offset=offset,
                axis_length=args.eef_axis_length,
                alpha=0.95,
            )

        if args.source in ("command", "both"):
            command_points = plot_arm_skeleton(
                ax_3d,
                kinematics,
                episode.command_joint_pos[frame_idx, arm_slice],
                color=arm_color,
                offset=offset,
                linestyle="--",
                alpha=0.65,
                linewidth=1.8,
            )
            plot_eef_trajectory(
                ax_3d,
                episode.command_eef_pose[:, eef_slice],
                frame_idx=frame_idx,
                color=arm_color,
                offset=offset,
                linestyle="--",
                alpha_full=0.08,
                alpha_past=0.60,
                label=f"{arm_name} command",
            )
            plot_eef_axes(
                ax_3d,
                episode.command_eef_pose[frame_idx, eef_slice],
                offset=offset,
                axis_length=args.eef_axis_length * 0.85,
                alpha=0.65,
            )

    set_axes_equal(ax_3d, axes_points)
    ax_3d.set_title("Arm pose + EEF trajectory")
    ax_3d.set_xlabel("x [m]")
    ax_3d.set_ylabel("y [m]")
    ax_3d.set_zlabel("z [m]")
    ax_3d.grid(True, alpha=0.25)
    if args.source == "both":
        ax_3d.legend(loc="upper right")

    if args.arm in ("left", "both"):
        plot_joint_panel(ax_joint_left, episode, frame_idx, 0, "left", args)
        plot_eef_panel(ax_eef_left, episode, frame_idx, 0, "left", args)
        ax_eef_left.set_xlabel("time [s]")
    else:
        disable_axis(ax_joint_left, "Left joints")
        disable_axis(ax_eef_left, "Left EEF xyz")

    if args.arm in ("right", "both"):
        plot_joint_panel(ax_joint_right, episode, frame_idx, 1, "right", args)
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
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit("OpenCV is required for MP4 export. Install `opencv-python`.") from exc

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
    first_axes_points = gather_points_for_bounds(first_episode, first_indices, args, kinematics)
    probe_fig = render_episode_frame(
        plt,
        first_episode,
        frame_idx=int(first_indices[0]),
        frame_position=0,
        total_frames=len(first_indices),
        args=args,
        kinematics=kinematics,
        axes_points=first_axes_points,
    )
    probe_bgr = figure_to_bgr(probe_fig)
    plt.close(probe_fig)

    writer = open_video_writer(output_path, probe_bgr, output_fps)
    print(f"output: {output_path}")
    print(f"robot_urdf: {robot_urdf}")
    print(f"episodes: {len(episode_sources)}")
    print(f"video_fps: {output_fps:.3f}")

    try:
        for episode_index, source in enumerate(episode_sources):
            episode = first_episode if episode_index == 0 else load_episode_data(source)
            frame_indices = build_frame_indices(len(episode.time_values), args)
            if len(frame_indices) == 0:
                print(f"⚠️  Skipping empty frame selection: {source.hdf5_path}")
                continue

            axes_points = first_axes_points if episode_index == 0 else gather_points_for_bounds(episode, frame_indices, args, kinematics)
            if len(frame_indices) > 500 and args.max_frames_per_episode is None:
                print(
                    f"⚠️  {source.episode_name} will render {len(frame_indices)} frames. "
                    "Use --max_frames_per_episode to make previews faster."
                )

            print(
                f"🎬 Episode {episode_index + 1}/{len(episode_sources)}: {source.episode_name} "
                f"(frames={len(frame_indices)}, original={len(episode.time_values)})"
            )

            for local_index, frame_idx in enumerate(frame_indices):
                fig = render_episode_frame(
                    plt,
                    episode,
                    frame_idx=int(frame_idx),
                    frame_position=local_index,
                    total_frames=len(frame_indices),
                    args=args,
                    kinematics=kinematics,
                    axes_points=axes_points,
                )
                writer.write(figure_to_bgr(fig))
                plt.close(fig)
    finally:
        writer.release()

    print(f"saved: {output_path}")


def main():
    args = parse_args()
    episode_sources = resolve_episode_sources(args.input_path, args.max_episodes)
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else build_default_output_path(args.input_path, episode_sources)
    )
    robot_urdf = resolve_robot_urdf(args.robot_urdf)
    plt = load_matplotlib(args.backend)
    render_dataset_preview(args, plt, robot_urdf, episode_sources, output_path)


if __name__ == "__main__":
    main()
