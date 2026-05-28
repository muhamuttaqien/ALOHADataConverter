#!/usr/bin/env python3

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit("pyarrow is required to read LeRobot parquet episodes.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_to_rmb_from_lerobot import (
    compose_pose_delta,
    quaternion_to_rotation_matrix,
    relative_pose7_to_delta6,
    rotation_matrix_from_rotation_vector,
    rotation_vector_from_matrix,
)


ARM_SPECS = (
    ("left", "observation.pose.left_hand_root.absolute", "action.left_controller.relative", "#1f77b4"),
    ("right", "observation.pose.right_hand_root.absolute", "action.right_controller.relative", "#ff7f0e"),
)
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Candidate:
    perm: tuple[int, int, int]
    signs: tuple[int, int, int]
    translation_frame: str
    rotation_mode: str

    @property
    def axes_tokens(self):
        return tuple(("-" if sign < 0 else "") + AXES[index] for index, sign in zip(self.perm, self.signs))

    @property
    def label(self):
        perm_text = "".join(AXES[index] for index in self.perm)
        signs_text = "".join("+" if sign > 0 else "-" for sign in self.signs)
        return (
            f"axes={','.join(self.axes_tokens)} "
            f"perm={perm_text} signs={signs_text} trans={self.translation_frame} rot={self.rotation_mode}"
        )

    def to_dict(self):
        return {
            "perm": [AXES[index] for index in self.perm],
            "signs": list(self.signs),
            "axes": list(self.axes_tokens),
            "translation_frame": self.translation_frame,
            "rotation_mode": self.rotation_mode,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune coordinate conventions for LeRobot controller relative EEF actions by comparing "
            "one-step predictions against measured hand-root poses."
        )
    )
    parser.add_argument("--dataset_dir", required=True, help="LeRobot dataset directory.")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to inspect.")
    parser.add_argument("--top_k", type=int, default=12, help="Number of ranked candidates to print.")
    parser.add_argument(
        "--translation_frames",
        nargs="+",
        default=["world", "current_local", "current_local_inv"],
        choices=("world", "current_local", "current_local_inv"),
        help="Translation frame hypotheses to sweep.",
    )
    parser.add_argument(
        "--rotation_modes",
        nargs="+",
        default=["right", "left", "right_inv", "left_inv"],
        choices=("right", "left", "right_inv", "left_inv"),
        help="Rotation composition hypotheses to sweep.",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames used for scoring and rendering.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Stride for scoring and rendering.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON file for ranked scores.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional MP4 visualization for the best candidate.")
    parser.add_argument("--render_top", type=int, default=0, help="Render one MP4 per top-N candidate next to --output_path.")
    parser.add_argument("--fps", type=float, default=12.0, help="Output MP4 FPS.")
    parser.add_argument("--backend", type=str, default="Agg", help="Matplotlib backend for MP4 rendering.")
    parser.add_argument("--figure_scale", type=float, default=0.75, help="Rendered figure scale.")
    return parser.parse_args()


def find_episode_parquet(dataset_dir, episode_index):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    pattern = f"episode_{episode_index:06d}.parquet"
    matches = sorted(dataset_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} found under {dataset_dir}")
    return matches[0]


def list_column_to_array(table, column_name):
    if column_name not in table.column_names:
        raise KeyError(f"Missing Parquet column: {column_name}")
    return np.asarray(table[column_name].to_pylist(), dtype=np.float64)


def load_episode_arrays(parquet_path):
    table = pq.read_table(parquet_path)
    arrays = {}
    for arm_name, measured_column, relative_column, _ in ARM_SPECS:
        arrays[f"{arm_name}_measured"] = list_column_to_array(table, measured_column)
        arrays[f"{arm_name}_relative7"] = list_column_to_array(table, relative_column)
        arrays[f"{arm_name}_relative6"] = relative_pose7_to_delta6(arrays[f"{arm_name}_relative7"])
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64) if "timestamp" in table.column_names else None
    return arrays, timestamp


def build_frame_indices(num_frames, frame_stride, max_frames):
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    indices = np.arange(0, max(num_frames - 1, 0), frame_stride, dtype=np.int64)
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames)
        indices = indices[np.unique(np.round(positions).astype(np.int64))]
    return indices


def transform_delta(delta, current_pose, candidate):
    delta = np.asarray(delta, dtype=np.float64)
    current_rot = quaternion_to_rotation_matrix(current_pose[3:])

    xyz = delta[:3][list(candidate.perm)] * np.asarray(candidate.signs, dtype=np.float64)
    rotvec = delta[3:][list(candidate.perm)] * np.asarray(candidate.signs, dtype=np.float64)

    if candidate.translation_frame == "world":
        world_xyz = xyz
    elif candidate.translation_frame == "current_local":
        world_xyz = current_rot @ xyz
    elif candidate.translation_frame == "current_local_inv":
        world_xyz = current_rot.T @ xyz
    else:
        raise ValueError(f"Unsupported translation frame: {candidate.translation_frame}")

    out = np.zeros(6, dtype=np.float64)
    out[:3] = world_xyz
    out[3:] = rotvec
    return out


def compose_with_rotation_mode(pose, delta, rotation_mode):
    if rotation_mode == "right":
        return compose_pose_delta(pose, delta)

    pose = np.asarray(pose, dtype=np.float64)
    base_rot = quaternion_to_rotation_matrix(pose[3:])
    delta_rot = rotation_matrix_from_rotation_vector(delta[3:])
    if rotation_mode == "left":
        out_rot = delta_rot @ base_rot
    elif rotation_mode == "right_inv":
        out_rot = base_rot @ delta_rot.T
    elif rotation_mode == "left_inv":
        out_rot = delta_rot.T @ base_rot
    else:
        raise ValueError(f"Unsupported rotation mode: {rotation_mode}")

    out = np.zeros(7, dtype=np.float64)
    out[:3] = pose[:3] + delta[:3]
    out[3:] = quaternion_from_rotation(out_rot)
    if np.dot(out[3:], pose[3:]) < 0.0:
        out[3:] *= -1.0
    return out


def quaternion_from_rotation(rotation):
    # Imported converter exposes quaternion_xyzw_from_rotation indirectly through compose_pose_delta only.
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if np.isclose(norm, 0.0) else quat / norm


def predict_next_pose(current_pose, relative_delta6, candidate):
    delta = transform_delta(relative_delta6, current_pose, candidate)
    return compose_with_rotation_mode(current_pose, delta, candidate.rotation_mode)


def pose_errors(predicted, target):
    pos_error = np.linalg.norm(predicted[:3] - target[:3])
    pred_rot = quaternion_to_rotation_matrix(predicted[3:])
    target_rot = quaternion_to_rotation_matrix(target[3:])
    rot_error = np.linalg.norm(rotation_vector_from_matrix(pred_rot.T @ target_rot))
    return pos_error, rot_error


def candidate_grid(translation_frames, rotation_modes):
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            for translation_frame in translation_frames:
                for rotation_mode in rotation_modes:
                    yield Candidate(perm=perm, signs=signs, translation_frame=translation_frame, rotation_mode=rotation_mode)


def score_candidate(arrays, frame_indices, candidate):
    pos_errors = []
    rot_errors = []
    for arm_name, _, _, _ in ARM_SPECS:
        measured = arrays[f"{arm_name}_measured"]
        relative = arrays[f"{arm_name}_relative6"]
        for frame_idx in frame_indices:
            predicted = predict_next_pose(measured[frame_idx], relative[frame_idx], candidate)
            pos_error, rot_error = pose_errors(predicted, measured[frame_idx + 1])
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)

    pos_errors = np.asarray(pos_errors, dtype=np.float64)
    rot_errors = np.asarray(rot_errors, dtype=np.float64)
    return {
        "candidate": candidate,
        "pos_median": float(np.median(pos_errors)),
        "pos_mean": float(np.mean(pos_errors)),
        "pos_p95": float(np.percentile(pos_errors, 95)),
        "pos_max": float(np.max(pos_errors)),
        "rot_median": float(np.median(rot_errors)),
        "rot_mean": float(np.mean(rot_errors)),
        "rot_p95": float(np.percentile(rot_errors, 95)),
        "rot_max": float(np.max(rot_errors)),
    }


def integrate_candidate_trajectory(measured, relative, candidate):
    predicted = np.zeros_like(measured)
    if len(measured) == 0:
        return predicted
    predicted[0] = measured[0]
    for index in range(len(measured) - 1):
        predicted[index + 1] = predict_next_pose(predicted[index], relative[index], candidate)
    return predicted


def load_matplotlib(backend):
    import matplotlib

    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    return plt


def load_cv2():
    import cv2

    return cv2


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).max() * 0.55, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def figure_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[:, :, :3][:, :, ::-1].copy()


def render_candidate(output_path, arrays, frame_indices, candidate, score, args):
    plt = load_matplotlib(args.backend)
    cv2 = load_cv2()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_by_arm = {
        arm_name: integrate_candidate_trajectory(arrays[f"{arm_name}_measured"], arrays[f"{arm_name}_relative6"], candidate)
        for arm_name, _, _, _ in ARM_SPECS
    }
    points = []
    for arm_name, _, _, _ in ARM_SPECS:
        points.append(arrays[f"{arm_name}_measured"][:, :3])
        points.append(predicted_by_arm[arm_name][:, :3])
    points = np.concatenate(points, axis=0)

    probe = render_frame(plt, arrays, predicted_by_arm, frame_indices[0], candidate, score, points, args)
    probe_bgr = figure_to_bgr(probe)
    height, width = probe_bgr.shape[:2]
    plt.close(probe)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer: {output_path}")
    try:
        for frame_idx in frame_indices:
            fig = render_frame(plt, arrays, predicted_by_arm, int(frame_idx), candidate, score, points, args)
            writer.write(figure_to_bgr(fig))
            plt.close(fig)
    finally:
        writer.release()
    print(f"saved: {output_path}")


def render_frame(plt, arrays, predicted_by_arm, frame_idx, candidate, score, axes_points, args):
    scale = max(float(args.figure_scale), 0.2)
    fig = plt.figure(figsize=(16 * scale, 9.5 * scale), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    axx = fig.add_subplot(grid[0, 1])
    axy = fig.add_subplot(grid[1, 1])

    title = (
        f"{candidate.label}\n"
        f"pos median/p95/max={score['pos_median']:.4f}/{score['pos_p95']:.4f}/{score['pos_max']:.4f} m | "
        f"rot median={score['rot_median']:.4f} rad | frame={frame_idx}"
    )
    fig.suptitle(title, fontsize=11)

    for arm_name, _, _, color in ARM_SPECS:
        measured = arrays[f"{arm_name}_measured"]
        predicted = predicted_by_arm[arm_name]
        ax3d.plot(measured[:, 0], measured[:, 1], measured[:, 2], color=color, linewidth=1.4, alpha=0.35)
        ax3d.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], color=color, linestyle="--", linewidth=1.2, alpha=0.35)
        ax3d.plot(measured[: frame_idx + 1, 0], measured[: frame_idx + 1, 1], measured[: frame_idx + 1, 2], color=color, linewidth=2.0, label=f"{arm_name} measured")
        ax3d.plot(predicted[: frame_idx + 1, 0], predicted[: frame_idx + 1, 1], predicted[: frame_idx + 1, 2], color=color, linestyle="--", linewidth=2.0, label=f"{arm_name} predicted")
        ax3d.scatter([measured[frame_idx, 0]], [measured[frame_idx, 1]], [measured[frame_idx, 2]], color=color, s=40)
        ax3d.scatter([predicted[frame_idx, 0]], [predicted[frame_idx, 1]], [predicted[frame_idx, 2]], color=color, marker="x", s=45)

        t = np.arange(len(measured))
        for axis_index, axis_name in enumerate(AXES):
            ax = axx if axis_index < 2 else axy
            alpha = 0.9 if axis_index == 0 else 0.55
            ax.plot(t, measured[:, axis_index], color=color, linewidth=1.2, alpha=alpha, label=f"{arm_name} {axis_name} measured" if axis_index == 0 else None)
            ax.plot(t, predicted[:, axis_index], color=color, linestyle="--", linewidth=1.0, alpha=alpha, label=f"{arm_name} {axis_name} predicted" if axis_index == 0 else None)
            ax.scatter([frame_idx], [measured[frame_idx, axis_index]], color=color, s=12)
            ax.scatter([frame_idx], [predicted[frame_idx, axis_index]], color=color, marker="x", s=18)

    set_axes_equal(ax3d, axes_points)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.grid(True, alpha=0.25)
    ax3d.legend(loc="upper right", fontsize=8)
    axx.set_title("x/y over frames")
    axy.set_title("z over frames")
    for ax in (axx, axy):
        ax.axvline(frame_idx, color="#111111", linewidth=1.0, alpha=0.45)
        ax.grid(True, alpha=0.25)
    return fig


def write_json(path, rankings):
    serializable = []
    for rank, row in enumerate(rankings, start=1):
        payload = {key: value for key, value in row.items() if key != "candidate"}
        payload["rank"] = rank
        payload["candidate"] = row["candidate"].to_dict()
        payload["label"] = row["candidate"].label
        serializable.append(payload)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"wrote: {path}")


def main():
    args = parse_args()
    parquet_path = find_episode_parquet(args.dataset_dir, args.episode_index)
    arrays, _ = load_episode_arrays(parquet_path)
    num_frames = min(len(arrays["left_measured"]), len(arrays["right_measured"]))
    frame_indices = build_frame_indices(num_frames, args.frame_stride, args.max_frames)
    if len(frame_indices) == 0:
        raise ValueError("No frames to score.")

    rankings = [
        score_candidate(arrays, frame_indices, candidate)
        for candidate in candidate_grid(args.translation_frames, args.rotation_modes)
    ]
    rankings.sort(key=lambda row: (row["pos_median"], row["pos_p95"], row["rot_median"]))

    print(f"parquet: {parquet_path}")
    print(f"frames scored: {len(frame_indices)} / {num_frames}")
    for rank, row in enumerate(rankings[: args.top_k], start=1):
        print(
            f"{rank:02d} {row['candidate'].label} | "
            f"pos median/p95/max={row['pos_median']:.6f}/{row['pos_p95']:.6f}/{row['pos_max']:.6f} m | "
            f"rot median/p95={row['rot_median']:.6f}/{row['rot_p95']:.6f} rad"
        )

    if args.output_json:
        write_json(args.output_json, rankings)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        render_count = max(args.render_top, 1)
        for rank, row in enumerate(rankings[:render_count], start=1):
            if render_count == 1:
                candidate_path = output_path
            else:
                candidate_path = output_path.with_name(f"{output_path.stem}_rank{rank:02d}{output_path.suffix}")
            render_candidate(candidate_path, arrays, frame_indices, row["candidate"], row, args)


if __name__ == "__main__":
    main()
