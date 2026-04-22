#!/usr/bin/env python3

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
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
ARM_JOINT_DIM = 7
ARM_DOF_FOR_FK = 6
ARM_SLICES = (
    slice(0, ARM_JOINT_DIM),
    slice(ARM_JOINT_DIM, 2 * ARM_JOINT_DIM),
)
DEFAULT_URDF_CANDIDATES = (
    Path(__file__).resolve().parent / "config" / "vx300s.urdf",
    Path(__file__).resolve().parent / "assets" / "vx300s.urdf",
    Path.cwd() / "config" / "vx300s.urdf",
)


def translation_matrix(xyz):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def rotation_matrix_from_rpy(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def homogeneous_from_origin(xyz, rpy):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_rpy(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if np.isclose(norm, 0.0):
        return np.eye(4, dtype=np.float64)

    axis = axis / norm
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def quaternion_xyzw_from_rotation(rotation):
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
    if np.isclose(norm, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def rotation_vector_from_matrix(rotation):
    trace = np.trace(rotation)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0.0):
        return np.zeros(3, dtype=np.float64)

    if np.isclose(theta, np.pi):
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = np.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        norm = np.linalg.norm(axis)
        if np.isclose(norm, 0.0):
            return np.zeros(3, dtype=np.float64)
        return axis / norm * theta

    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )


class UrdfKinematics:
    def __init__(self, urdf_path, target_link="vx300s/ee_gripper_link"):
        self.urdf_path = Path(urdf_path)
        self.target_link = target_link
        self.chain = self._build_chain()
        self.num_actuated_joints = sum(1 for joint in self.chain if joint["type"] != "fixed")

    def _build_chain(self):
        root = ET.parse(self.urdf_path).getroot()
        joints_by_child = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
            axis = joint.find("axis")
            axis_xyz = [float(v) for v in axis.get("xyz", "0 0 0").split()] if axis is not None else [0.0, 0.0, 0.0]
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            joints_by_child[child.get("link")] = {
                "name": joint.get("name", ""),
                "type": joint.get("type", "fixed"),
                "parent": parent.get("link"),
                "child": child.get("link"),
                "origin_xyz": xyz,
                "origin_rpy": rpy,
                "axis": axis_xyz,
            }

        chain = []
        current_link = self.target_link
        while current_link in joints_by_child:
            joint = joints_by_child[current_link]
            chain.append(joint)
            current_link = joint["parent"]

        if not chain:
            raise ValueError(f"Failed to build URDF chain to target link: {self.target_link}")

        chain.reverse()
        return chain

    def forward_kinematics(self, joint_positions):
        return self.forward_kinematics_trace(joint_positions)[-1]["transform"]

    def forward_kinematics_trace(self, joint_positions):
        if len(joint_positions) < self.num_actuated_joints:
            raise ValueError(
                f"Expected at least {self.num_actuated_joints} joint positions for FK, got {len(joint_positions)}"
            )

        transform = np.eye(4, dtype=np.float64)
        trace = [
            {
                "name": "base",
                "type": "fixed",
                "transform": transform.copy(),
            }
        ]
        joint_index = 0
        for joint in self.chain:
            transform = transform @ homogeneous_from_origin(joint["origin_xyz"], joint["origin_rpy"])
            pre_motion_transform = transform.copy()
            if joint["type"] == "revolute" or joint["type"] == "continuous":
                transform = transform @ axis_angle_matrix(joint["axis"], joint_positions[joint_index])
                joint_index += 1
            elif joint["type"] == "prismatic":
                transform = transform @ translation_matrix(np.asarray(joint["axis"], dtype=np.float64) * joint_positions[joint_index])
                joint_index += 1
            trace.append(
                {
                    "name": joint["name"],
                    "type": joint["type"],
                    "axis": np.asarray(joint["axis"], dtype=np.float64),
                    "pre_motion_transform": pre_motion_transform,
                    "transform": transform.copy(),
                }
            )
        return trace

def resolve_robot_urdf(robot_urdf):
    if robot_urdf is not None:
        path = Path(robot_urdf).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Robot URDF not found: {path}")
        return path

    for candidate in DEFAULT_URDF_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    candidates = ", ".join(str(path) for path in DEFAULT_URDF_CANDIDATES)
    raise FileNotFoundError(f"Robot URDF not found. Checked: {candidates}")


def compute_eef_pose_sequence(joint_sequence, kinematics):
    joint_sequence = np.asarray(joint_sequence, dtype=np.float64)
    poses = np.zeros((joint_sequence.shape[0], 7), dtype=np.float64)
    for index, joints in enumerate(joint_sequence):
        transform = kinematics.forward_kinematics(joints[:ARM_DOF_FOR_FK])
        poses[index, :3] = transform[:3, 3]
        poses[index, 3:] = quaternion_xyzw_from_rotation(transform[:3, :3])
    return poses


def compute_eef_pose_rel(poses):
    poses = np.asarray(poses, dtype=np.float64)
    rel = np.zeros((poses.shape[0], 6), dtype=np.float64)
    for index in range(1, poses.shape[0]):
        rel[index, :3] = poses[index, :3] - poses[index - 1, :3]
        prev_rot = quaternion_to_rotation_matrix(poses[index - 1, 3:])
        curr_rot = quaternion_to_rotation_matrix(poses[index, 3:])
        rel[index, 3:] = rotation_vector_from_matrix(prev_rot.T @ curr_rot)
    return rel


def quaternion_to_rotation_matrix(quat_xyzw):
    qx, qy, qz, qw = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm([qx, qy, qz, qw])
    if np.isclose(norm, 0.0):
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = np.asarray([qx, qy, qz, qw], dtype=np.float64) / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def build_bimanual_eef_data(joint_sequence, kinematics):
    poses_per_arm = []
    rel_per_arm = []
    for arm_slice in ARM_SLICES:
        arm_joints = joint_sequence[:, arm_slice]
        arm_poses = compute_eef_pose_sequence(arm_joints, kinematics)
        poses_per_arm.append(arm_poses)
        rel_per_arm.append(compute_eef_pose_rel(arm_poses))
    return np.concatenate(poses_per_arm, axis=1), np.concatenate(rel_per_arm, axis=1)


def build_bimanual_eef_wrench_data(effort_sequence):
    effort_sequence = np.asarray(effort_sequence, dtype=np.float64)
    wrench_per_arm = []
    for arm_slice in ARM_SLICES:
        arm_effort = effort_sequence[:, arm_slice]
        wrench_per_arm.append(arm_effort[:, :ARM_DOF_FOR_FK])
    return np.concatenate(wrench_per_arm, axis=1)


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


def save_episode_hdf5(
    out_path,
    task_desc,
    qpos,
    qvel,
    action,
    time_values,
    camera_names,
    kinematics,
    effort=None,
):
    dtype_f8 = np.float64
    num_steps = len(time_values)
    num_grippers = len(GRIPPER_INDICES)
    num_eef = num_grippers

    if qpos.shape[1] < ARM_SLICES[-1].stop or action.shape[1] < ARM_SLICES[-1].stop:
        raise ValueError(
            f"Expected at least {ARM_SLICES[-1].stop} joint dimensions for bimanual FK, "
            f"got qpos={qpos.shape[1]}, action={action.shape[1]}"
        )

    measured_joint_pos = qpos.astype(dtype_f8)
    measured_joint_vel = qvel.astype(dtype_f8)
    command_joint_pos = action.astype(dtype_f8)

    measured_gripper_joint_pos = qpos[:, GRIPPER_INDICES].astype(dtype_f8)
    command_gripper_joint_pos = action[:, GRIPPER_INDICES].astype(dtype_f8)

    measured_joint_pos_rel = previous_step_delta(measured_joint_pos)
    command_joint_pos_rel = previous_step_delta(command_joint_pos)
    measured_gripper_joint_pos_rel = previous_step_delta(measured_gripper_joint_pos)
    command_gripper_joint_pos_rel = previous_step_delta(command_gripper_joint_pos)
    measured_eef_pose, measured_eef_pose_rel = build_bimanual_eef_data(measured_joint_pos, kinematics)
    command_eef_pose, command_eef_pose_rel = build_bimanual_eef_data(command_joint_pos, kinematics)
    if effort is not None:
        effort = np.asarray(effort, dtype=dtype_f8)
        if effort.shape[1] < ARM_SLICES[-1].stop:
            raise ValueError(
                f"Expected at least {ARM_SLICES[-1].stop} effort dimensions for bimanual wrench, got {effort.shape[1]}"
            )
        measured_eef_wrench = build_bimanual_eef_wrench_data(effort)
    else:
        measured_eef_wrench = np.zeros((num_steps, 6 * num_eef), dtype=dtype_f8)
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


def process_single_hdf5(args):
    hdf5_file, dataset_name, out_dir, fps, camera_workers, video_preset, robot_urdf = args

    episode_name = episode_output_name(hdf5_file)
    rmb_dir = out_dir / dataset_name / episode_name
    rmb_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Reading file: {hdf5_file}")

    with h5py.File(hdf5_file, "r") as root:
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        effort = root["/observations/effort"][()] if "/observations/effort" in root else None
        action = root["/action"][()]
        task_desc = extract_task_desc(root)
        source_fps = infer_source_fps(root, fps)

        sample_indices = build_sample_indices(len(qpos), source_fps, fps)
        qpos_rs = qpos[sample_indices]
        qvel_rs = qvel[sample_indices]
        effort_rs = effort[sample_indices] if effort is not None else None
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
        effort=effort_rs,
        time_values=time_values,
        camera_names=exported_camera_names,
        kinematics=UrdfKinematics(robot_urdf),
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


def process_dataset(
    input_path,
    out_dir,
    fps=25,
    nproc=1,
    camera_workers=0,
    video_preset="veryfast",
    robot_urdf=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_urdf = resolve_robot_urdf(robot_urdf)

    dataset_folders = iter_dataset_folders(input_path)
    if not dataset_folders:
        print(f"❌ No HDF5 episodes found under: {input_path}")
        return

    for dataset_name, hdf5_files in dataset_folders:
        print(f"\n📦 Processing folder: {dataset_name}")
        args_list = [
            (hdf5_file, dataset_name, out_dir, fps, camera_workers, video_preset, robot_urdf)
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
    parser.add_argument(
        "--robot_urdf",
        type=str,
        default=None,
        help="Path to the single-arm robot URDF used to compute EEF poses. Defaults to config/assets vx300s URDF.",
    )
    args = parser.parse_args()

    process_dataset(
        input_path=args.input_dir,
        out_dir=args.output_dir,
        fps=args.fps,
        nproc=args.nproc,
        camera_workers=args.camera_workers,
        video_preset=args.video_preset,
        robot_urdf=args.robot_urdf,
    )


if __name__ == "__main__":
    main()
