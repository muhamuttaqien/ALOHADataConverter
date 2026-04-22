#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError as exc:
    raise SystemExit("h5py is required to read ALOHA episodes.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_to_rmb import ARM_DOF_FOR_FK, ARM_SLICES, quaternion_xyzw_from_rotation, resolve_robot_urdf, UrdfKinematics


ARM_LABELS = (
    ("left", "#1f77b4", np.array([0.0, 0.30, 0.0], dtype=np.float64)),
    ("right", "#ff7f0e", np.array([0.0, -0.30, 0.0], dtype=np.float64)),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VX300S FK from ALOHA qpos/action data.")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to an input episode HDF5 file.")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument(
        "--robot_urdf",
        type=str,
        default=None,
        help="Path to the VX300S URDF. Defaults to config/assets vx300s.urdf.",
    )
    parser.add_argument(
        "--source",
        choices=("qpos", "action", "both"),
        default="both",
        help="Which joint source to draw.",
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="both",
        help="Which arm to visualize.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional output image path. Defaults to misc/fk_frame_<index>.png next to the HDF5 file.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Optional MP4 output path. If set, render a frame range as video instead of a single PNG.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    parser.add_argument(
        "--eef_axis_length",
        type=float,
        default=0.06,
        help="Length in meters for the EEF orientation axes.",
    )
    parser.add_argument("--frame_start", type=int, default=0, help="Start frame index for MP4 export.")
    parser.add_argument("--frame_stop", type=int, default=None, help="Exclusive stop frame index for MP4 export.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride for MP4 export.")
    parser.add_argument("--fps", type=float, default=20.0, help="FPS for MP4 export.")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional matplotlib backend. If omitted, the script uses Agg unless interactive display is requested.",
    )
    return parser.parse_args()


def load_episode(hdf5_path):
    with h5py.File(hdf5_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
    return qpos, action


def load_frame(hdf5_path, frame_idx):
    qpos, action = load_episode(hdf5_path)

    if len(qpos) == 0:
        raise ValueError(f"No frames in {hdf5_path}")
    if frame_idx < 0 or frame_idx >= len(qpos):
        raise IndexError(f"frame_idx out of range: {frame_idx} not in [0, {len(qpos) - 1}]")

    return qpos[frame_idx], action[frame_idx], len(qpos)


def transform_to_pose(transform):
    pose = np.zeros(7, dtype=np.float64)
    pose[:3] = transform[:3, 3]
    pose[3:] = quaternion_xyzw_from_rotation(transform[:3, :3])
    return pose


def plot_eef_axes(ax, transform, offset, axis_length):
    origin = transform[:3, 3] + offset
    rotation = transform[:3, :3]
    colors = ("#d62728", "#2ca02c", "#9467bd")
    for axis_index, color in enumerate(colors):
        tip = origin + rotation[:, axis_index] * axis_length
        ax.plot(
            [origin[0], tip[0]],
            [origin[1], tip[1]],
            [origin[2], tip[2]],
            color=color,
            linewidth=2.0,
        )


def plot_arm(ax, kinematics, arm_name, joints, style_name, color, offset, axis_length):
    trace = kinematics.forward_kinematics_trace(joints[:ARM_DOF_FOR_FK])
    points = np.array([entry["transform"][:3, 3] + offset for entry in trace], dtype=np.float64)
    linestyle = "-" if style_name == "qpos" else "--"
    alpha = 0.95 if style_name == "qpos" else 0.65
    linewidth = 2.5 if style_name == "qpos" else 1.8
    label = f"{arm_name} {style_name}"
    ax.plot(points[:, 0], points[:, 1], points[:, 2], linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha, label=label)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=22, alpha=alpha)
    plot_eef_axes(ax, trace[-1]["transform"], offset, axis_length)
    return transform_to_pose(trace[-1]["transform"]), points


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).max() * 0.55, 0.15)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def build_default_save_path(hdf5_path, frame_idx):
    hdf5_path = Path(hdf5_path)
    return hdf5_path.parent / f"fk_frame_{frame_idx:04d}.png"


def build_default_video_path(hdf5_path, frame_start, frame_stop):
    hdf5_path = Path(hdf5_path)
    return hdf5_path.parent / f"fk_frames_{frame_start:04d}_{frame_stop:04d}.mp4"


def load_matplotlib(show, backend):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for visualization. Install it with `pip install matplotlib`.") from exc

    selected_backend = backend
    if selected_backend is None and not show:
        selected_backend = "Agg"

    if selected_backend is not None:
        matplotlib.use(selected_backend)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"Failed to initialize matplotlib backend '{selected_backend or 'default'}': {exc}") from exc

    return plt


def gather_frame_points(args, kinematics, qpos_frame, action_frame):
    all_points = []
    for arm_index, (arm_name, _, offset) in enumerate(ARM_LABELS):
        if args.arm != "both" and args.arm != arm_name:
            continue
        arm_slice = ARM_SLICES[arm_index]
        if args.source in ("qpos", "both"):
            trace = kinematics.forward_kinematics_trace(qpos_frame[arm_slice][:ARM_DOF_FOR_FK])
            all_points.extend(entry["transform"][:3, 3] + offset for entry in trace)
        if args.source in ("action", "both"):
            trace = kinematics.forward_kinematics_trace(action_frame[arm_slice][:ARM_DOF_FOR_FK])
            all_points.extend(entry["transform"][:3, 3] + offset for entry in trace)
    if not all_points:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return np.array(all_points, dtype=np.float64)


def compute_global_points(args, kinematics, qpos_all, action_all, frame_indices):
    points = []
    for frame_idx in frame_indices:
        points.extend(gather_frame_points(args, kinematics, qpos_all[frame_idx], action_all[frame_idx]))
    if not points:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return np.array(points, dtype=np.float64)


def render_figure(
    plt,
    args,
    hdf5_path,
    robot_urdf,
    kinematics,
    qpos_frame,
    action_frame,
    num_frames,
    axes_points=None,
    log_prefix="",
    show_context=True,
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []

    if show_context:
        print(f"{log_prefix}HDF5: {hdf5_path}")
        print(f"{log_prefix}URDF: {robot_urdf}")
    print(f"{log_prefix}frame_idx: {args.frame_idx} / {num_frames - 1}")

    for arm_index, (arm_name, color, offset) in enumerate(ARM_LABELS):
        if args.arm != "both" and args.arm != arm_name:
            continue

        arm_slice = ARM_SLICES[arm_index]

        if args.source in ("qpos", "both"):
            pose, points = plot_arm(ax, kinematics, arm_name, qpos_frame[arm_slice], "qpos", color, offset, args.eef_axis_length)
            all_points.extend(points)
            print(f"{log_prefix}{arm_name} qpos eef pose: {np.array2string(pose, precision=6, suppress_small=True)}")

        if args.source in ("action", "both"):
            pose, points = plot_arm(ax, kinematics, arm_name, action_frame[arm_slice], "action", color, offset, args.eef_axis_length * 0.85)
            all_points.extend(points)
            print(f"{log_prefix}{arm_name} action eef pose: {np.array2string(pose, precision=6, suppress_small=True)}")

    all_points = np.array(all_points if all_points else [[0.0, 0.0, 0.0]], dtype=np.float64)
    set_axes_equal(ax, axes_points if axes_points is not None else all_points)

    ax.set_title(f"ALOHA FK visualization: {hdf5_path.name} frame {args.frame_idx}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def figure_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[:, :, :3]
    return rgb[:, :, ::-1].copy()


def export_video(args, plt, hdf5_path, robot_urdf, kinematics, qpos_all, action_all):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit("OpenCV is required for MP4 export. Install `opencv-python`.") from exc

    num_frames = len(qpos_all)
    frame_start = max(0, args.frame_start)
    frame_stop = num_frames if args.frame_stop is None else min(args.frame_stop, num_frames)
    if frame_stop <= frame_start:
        raise ValueError(f"Invalid frame range: start={frame_start}, stop={frame_stop}")
    if args.frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {args.frame_stride}")
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    frame_indices = list(range(frame_start, frame_stop, args.frame_stride))
    axes_points = compute_global_points(args, kinematics, qpos_all, action_all, frame_indices)
    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else build_default_video_path(hdf5_path, frame_start, frame_stop)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"HDF5: {hdf5_path}")
    print(f"URDF: {robot_urdf}")
    print(f"video frames: {frame_indices[0]}..{frame_indices[-1]} (count={len(frame_indices)}, stride={args.frame_stride}, fps={args.fps})")

    probe_args = argparse.Namespace(**vars(args))
    probe_args.frame_idx = frame_indices[0]
    probe_fig = render_figure(
        plt,
        probe_args,
        hdf5_path,
        robot_urdf,
        kinematics,
        qpos_all[frame_indices[0]],
        action_all[frame_indices[0]],
        num_frames,
        axes_points=axes_points,
        log_prefix="[probe] ",
        show_context=False,
    )
    probe_bgr = figure_to_bgr(probe_fig)
    height, width = probe_bgr.shape[:2]
    plt.close(probe_fig)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {video_path}")

    try:
        for index, frame_idx in enumerate(frame_indices):
            frame_args = argparse.Namespace(**vars(args))
            frame_args.frame_idx = frame_idx
            fig = render_figure(
                plt,
                frame_args,
                hdf5_path,
                robot_urdf,
                kinematics,
                qpos_all[frame_idx],
                action_all[frame_idx],
                num_frames,
                axes_points=axes_points,
                log_prefix=f"[frame {index + 1}/{len(frame_indices)}] ",
                show_context=False,
            )
            writer.write(figure_to_bgr(fig))
            plt.close(fig)
    finally:
        writer.release()

    print(f"saved: {video_path}")


def main():
    args = parse_args()

    if args.video_path is not None:
        args.show = False

    if args.show and args.backend is None and not os.environ.get("DISPLAY"):
        print("DISPLAY is not set; using Agg backend and saving the figure without opening a window.", file=sys.stderr)
        args.show = False

    plt = load_matplotlib(show=args.show, backend=args.backend)

    hdf5_path = Path(args.hdf5).expanduser().resolve()
    robot_urdf = resolve_robot_urdf(args.robot_urdf)
    kinematics = UrdfKinematics(robot_urdf)
    qpos_all, action_all = load_episode(hdf5_path)
    num_frames = len(qpos_all)

    if num_frames == 0:
        raise ValueError(f"No frames in {hdf5_path}")

    if args.video_path is not None:
        export_video(args, plt, hdf5_path, robot_urdf, kinematics, qpos_all, action_all)
        return

    if args.frame_idx < 0 or args.frame_idx >= num_frames:
        raise IndexError(f"frame_idx out of range: {args.frame_idx} not in [0, {num_frames - 1}]")
    qpos_frame = qpos_all[args.frame_idx]
    action_frame = action_all[args.frame_idx]

    try:
        fig = render_figure(plt, args, hdf5_path, robot_urdf, kinematics, qpos_frame, action_frame, num_frames)
    except Exception as exc:
        if args.show and args.backend is None:
            print(
                f"Interactive backend initialization failed ({exc}). Falling back to Agg and saving only.",
                file=sys.stderr,
            )
            plt.switch_backend("Agg")
            args.show = False
            fig = render_figure(plt, args, hdf5_path, robot_urdf, kinematics, qpos_frame, action_frame, num_frames)
        else:
            raise

    save_path = Path(args.save_path).expanduser().resolve() if args.save_path else build_default_save_path(hdf5_path, args.frame_idx)
    fig.savefig(save_path, dpi=180)
    print(f"saved: {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
