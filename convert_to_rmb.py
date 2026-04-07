#!/usr/bin/env python3

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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
    import cv2
except ModuleNotFoundError:
    _extend_import_paths()
    import cv2

try:
    import h5py
except ModuleNotFoundError:
    _extend_import_paths()
    import h5py

try:
    import videoio
except ModuleNotFoundError:
    _extend_import_paths()
    import videoio

import numpy as np

try:
    import natsort
except ModuleNotFoundError:
    natsort = None


GRIPPER_INDICES = (6, 13)


def natsorted_paths(paths):
    if natsort is not None:
        return list(natsort.natsorted(paths))
    return sorted(paths, key=lambda path: path.name)


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

    return ""


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


def previous_step_delta(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy()
    prev = np.vstack([arr[:1], arr[:-1]])
    return arr - prev


def episode_output_name(hdf5_file):
    match = re.search(r"(\d+)$", hdf5_file.stem)
    if match:
        return f"episode_{int(match.group(1)):06d}.rmb"
    return f"{hdf5_file.stem}.rmb"


def decode_rgb_frames(root, cam_name, sample_indices):
    dataset = root[f"/observations/images/{cam_name}"]
    frames = []
    for idx in sample_indices:
        frame_bgr = cv2.imdecode(dataset[idx], cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Failed to decode RGB frame from {cam_name}[{idx}]")
        frames.append(frame_bgr)
    return np.stack(frames, axis=0)


def convert_depth_frames(depth_frames, cam_name):
    if np.issubdtype(depth_frames.dtype, np.floating):
        return np.clip(np.round(depth_frames * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    if depth_frames.dtype == np.uint16:
        return depth_frames

    if np.issubdtype(depth_frames.dtype, np.integer):
        print(
            f"⚠️  Depth camera '{cam_name}' is stored as {depth_frames.dtype}; "
            "upcasting to uint16 for RMB, but units may not be millimeters."
        )
        return depth_frames.astype(np.uint16)

    raise TypeError(f"Unsupported depth dtype for {cam_name}: {depth_frames.dtype}")


def decode_depth_frames(root, cam_name, sample_indices):
    dataset = root[f"/observations/depth/{cam_name}"]
    sampled = dataset[sample_indices]
    return convert_depth_frames(sampled, cam_name)


def export_camera_stream(hdf5_file, sample_indices, fps, rmb_dir, cam_name, is_depth, video_preset):
    with h5py.File(hdf5_file, "r") as root:
        if is_depth:
            frames = decode_depth_frames(root, cam_name, sample_indices)
            video_path = rmb_dir / f"{cam_name}_depth_image.rmb.mp4"
            print(f"🎞️ Saving video: {video_path.name}")
            videoio.uint16save(video_path, frames, preset=video_preset, fps=fps)
        else:
            frames = decode_rgb_frames(root, cam_name, sample_indices)
            video_path = rmb_dir / f"{cam_name}_rgb_image.rmb.mp4"
            print(f"🎞️ Saving video: {video_path.name}")
            videoio.videosave(video_path, frames, lossless=False, preset=video_preset, fps=fps)


def save_episode_hdf5(out_path, task_desc, qpos, qvel, action, time_values, camera_names):
    dtype_f8 = np.float64
    num_steps = len(time_values)
    num_grippers = len(GRIPPER_INDICES)
    num_eef = num_grippers

    measured_joint_pos = qpos.astype(dtype_f8)
    measured_joint_vel = qvel.astype(dtype_f8)
    command_joint_pos = action.astype(dtype_f8)

    measured_gripper_joint_pos = qpos[:, GRIPPER_INDICES].astype(dtype_f8)
    command_gripper_joint_pos = action[:, GRIPPER_INDICES].astype(dtype_f8)

    measured_joint_pos_rel = previous_step_delta(measured_joint_pos)
    command_joint_pos_rel = previous_step_delta(command_joint_pos)
    measured_gripper_joint_pos_rel = previous_step_delta(measured_gripper_joint_pos)
    command_gripper_joint_pos_rel = previous_step_delta(command_gripper_joint_pos)

    zeros_pose = np.zeros((num_steps, 7 * num_eef), dtype=dtype_f8)
    zeros_pose_rel = np.zeros((num_steps, 6 * num_eef), dtype=dtype_f8)
    zeros_wrench = np.zeros((num_steps, 6 * num_eef), dtype=dtype_f8)
    zeros_reward = np.zeros((num_steps,), dtype=dtype_f8)

    with h5py.File(out_path, "w") as f:
        f.attrs["camera_names"] = np.array(camera_names, dtype=object)
        f.attrs["demo_name"] = "AlohaRmbDemo"
        f.attrs["env"] = "AlohaRmbDemoEnv"
        f.attrs["format"] = "RmbData-Compact"
        f.attrs["task_desc"] = task_desc
        f.attrs["version"] = "3.0.0"
        f.attrs["world_idx"] = 0
        f.attrs["pointcloud_camera_names"] = np.array([], dtype=np.float64)
        f.attrs["rgb_tactile_names"] = np.array([], dtype=np.float64)

        f.create_dataset("command_eef_pose", data=zeros_pose)
        f.create_dataset("command_eef_pose_rel", data=zeros_pose_rel)
        f.create_dataset("command_gripper_joint_pos", data=command_gripper_joint_pos)
        f.create_dataset("command_gripper_joint_pos_rel", data=command_gripper_joint_pos_rel)
        f.create_dataset("command_joint_pos", data=command_joint_pos)
        f.create_dataset("command_joint_pos_rel", data=command_joint_pos_rel)
        f.create_dataset("measured_eef_pose", data=zeros_pose)
        f.create_dataset("measured_eef_pose_rel", data=zeros_pose_rel)
        f.create_dataset("measured_eef_wrench", data=zeros_wrench)
        f.create_dataset("measured_gripper_joint_pos", data=measured_gripper_joint_pos)
        f.create_dataset("measured_gripper_joint_pos_rel", data=measured_gripper_joint_pos_rel)
        f.create_dataset("measured_joint_pos", data=measured_joint_pos)
        f.create_dataset("measured_joint_pos_rel", data=measured_joint_pos_rel)
        f.create_dataset("measured_joint_vel", data=measured_joint_vel)
        f.create_dataset("reward", data=zeros_reward)
        f.create_dataset("time", data=time_values.astype(dtype_f8))


def process_single_hdf5(args):
    hdf5_file, dataset_name, out_dir, fps, camera_workers, video_preset = args

    episode_name = episode_output_name(hdf5_file)
    rmb_dir = out_dir / dataset_name / episode_name
    rmb_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Reading file: {hdf5_file}")

    with h5py.File(hdf5_file, "r") as root:
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        action = root["/action"][()]
        task_desc = extract_task_desc(root)
        source_fps = infer_source_fps(root, fps)

        sample_indices = build_sample_indices(len(qpos), source_fps, fps)
        qpos_rs = qpos[sample_indices]
        qvel_rs = qvel[sample_indices]
        action_rs = action[sample_indices]
        time_values = sample_indices.astype(np.float64) / source_fps

        rgb_camera_names = list(root["/observations/images"].keys()) if "/observations/images" in root else []
        depth_camera_names = list(root["/observations/depth"].keys()) if "/observations/depth" in root else []
        exported_camera_names = rgb_camera_names + [name for name in depth_camera_names if name not in rgb_camera_names]

    camera_tasks = [(cam_name, False) for cam_name in rgb_camera_names] + [
        (cam_name, True) for cam_name in depth_camera_names
    ]

    if camera_tasks:
        max_workers = camera_workers
        if max_workers <= 0:
            max_workers = min(len(camera_tasks), os.cpu_count() or 1)

        if max_workers > 1 and len(camera_tasks) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        export_camera_stream,
                        hdf5_file,
                        sample_indices,
                        fps,
                        rmb_dir,
                        cam_name,
                        is_depth,
                        video_preset,
                    )
                    for cam_name, is_depth in camera_tasks
                ]
                for future in futures:
                    future.result()
        else:
            for cam_name, is_depth in camera_tasks:
                export_camera_stream(
                    hdf5_file,
                    sample_indices,
                    fps,
                    rmb_dir,
                    cam_name,
                    is_depth,
                    video_preset,
                )

    save_episode_hdf5(
        rmb_dir / "main.rmb.hdf5",
        task_desc=task_desc,
        qpos=qpos_rs,
        qvel=qvel_rs,
        action=action_rs,
        time_values=time_values,
        camera_names=exported_camera_names,
    )

    print(f"✅ Done: {episode_name}")


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


def process_dataset(input_path, out_dir, fps=25, nproc=1, camera_workers=0, video_preset="veryfast"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_folders = iter_dataset_folders(input_path)
    if not dataset_folders:
        print(f"❌ No HDF5 episodes found under: {input_path}")
        return

    for dataset_name, hdf5_files in dataset_folders:
        print(f"\n📦 Processing folder: {dataset_name}")
        args_list = [
            (hdf5_file, dataset_name, out_dir, fps, camera_workers, video_preset)
            for hdf5_file in hdf5_files
        ]

        if nproc > 1:
            with Pool(nproc) as pool:
                pool.map(process_single_hdf5, args_list)
        else:
            for args in args_list:
                process_single_hdf5(args)


def main():
    parser = argparse.ArgumentParser(description="Convert ALOHA HDF5 episodes into RMB-compatible output.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to an episode file, a dataset folder, or a root folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder.")
    parser.add_argument("--fps", type=float, default=25, help="Output video FPS.")
    parser.add_argument("--nproc", type=int, default=1, help="Number of parallel processes.")
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
    args = parser.parse_args()

    process_dataset(
        input_path=args.input_dir,
        out_dir=args.output_dir,
        fps=args.fps,
        nproc=args.nproc,
        camera_workers=args.camera_workers,
        video_preset=args.video_preset,
    )


if __name__ == "__main__":
    main()
