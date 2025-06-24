#!/usr/bin/env python3

import os
import h5py
import numpy as np
import argparse
import cv2
from pathlib import Path
import natsort
from multiprocessing import Pool

# Constants
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist"]
DCAMERA_NAMES = ["dcam_high", "dcam_low"]

def decode_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'❌ Dataset does not exist at {dataset_path}')
        exit(1)
    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]

        image_dict = {}
        for cam_name in root['/observations/images'].keys():
            emc_images = root[f'/observations/images/{cam_name}'][()]
            image_dict[cam_name] = []
            for img in emc_images:
                decompressed_image = cv2.imdecode(img, 1)
                image_dict[cam_name].append(decompressed_image)

    return is_sim, qpos, qvel, effort, action, image_dict

def save_images_to_video(images, out_path, fps=30, is_depth=False):
    if images.ndim == 4:
        h, w = images.shape[1:3]
    else:
        h, w = images.shape[1:3]
        images = np.expand_dims(images, -1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)

    for frame in images:
        if is_depth:
            frame = np.squeeze(frame)
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = frame.astype(np.uint8)
        writer.write(frame)

    writer.release()

def save_episode_hdf5(out_path, is_sim, qpos, qvel, effort, action, fps):
    import h5py
    import numpy as np
    N = qpos.shape[0]

    # Utility for dtype
    dtype_f8 = '<f8'

    # Joint and gripper slices
    joint_slice = slice(0, 14)
    # gripper_slice = slice(6, 7)
    # gripper_slice = slice(12, 13)
    gripper_slice = slice(6, 7)

    with h5py.File(out_path, 'w') as f:
        f.attrs['sim'] = is_sim
        f.attrs['fps'] = fps

        # measured_joint_pos: (N, xxx)
        f.create_dataset('measured_joint_pos', data=qpos[:, joint_slice].astype(dtype_f8))
        # measured_joint_vel: (N, xxx)
        f.create_dataset('measured_joint_vel', data=qvel[:, joint_slice].astype(dtype_f8))
        # measured_gripper_joint_pos: (N, 1)
        f.create_dataset('measured_gripper_joint_pos', data=qpos[:, gripper_slice].astype(dtype_f8))
        # command_joint_pos: (N, xxx)
        f.create_dataset('command_joint_pos', data=action[:, joint_slice].astype(dtype_f8))
        # command_gripper_joint_pos: (N, 1)
        f.create_dataset('command_gripper_joint_pos', data=action[:, gripper_slice].astype(dtype_f8))

        # Empty arrays for eef and wrench
        f.create_dataset('measured_eef_pose', data=np.zeros((N, 7), dtype=dtype_f8))
        f.create_dataset('measured_eef_pose_rel', data=np.zeros((N, 6), dtype=dtype_f8))
        f.create_dataset('measured_eef_wrench', data=np.zeros((N, 6), dtype=dtype_f8))
        f.create_dataset('command_eef_pose', data=np.zeros((N, 7), dtype=dtype_f8))
        f.create_dataset('command_eef_pose_rel', data=np.zeros((N, 6), dtype=dtype_f8))

        # measured_joint_pos_rel: (N, xxx)
        measured_joint_pos_rel = qpos[:, joint_slice] - qpos[0, joint_slice]
        f.create_dataset('measured_joint_pos_rel', data=measured_joint_pos_rel.astype(dtype_f8))
        # command_joint_pos_rel: (N, xxx)
        command_joint_pos_rel = action[:, joint_slice] - action[0, joint_slice]
        f.create_dataset('command_joint_pos_rel', data=command_joint_pos_rel.astype(dtype_f8))

        # time: (N,)
        f.create_dataset('time', data=np.arange(N, dtype=dtype_f8)*1./fps)

def resample_array(arr, target_len):
    """Resample array to target length by uniform subsampling."""
    if len(arr) == target_len:
        return arr
    idxs = np.linspace(0, len(arr) - 1, target_len).astype(int)
    return arr[idxs]

def process_single_hdf5(args):
    hdf5_file, dataset_folder, out_dir, fps = args
    print(f"📄 Reading file: {hdf5_file.name}")
    episode_name = f"episode_{hdf5_file.stem.split('.')[0].replace('episode', 'episode_'):0>13}.rmb" if not hdf5_file.name.startswith("episode_") else f"{hdf5_file.stem}.rmb"
    rmb_dir = out_dir / dataset_folder.name / episode_name
    rmb_dir.mkdir(parents=True, exist_ok=True)

    is_sim, qpos, qvel, effort, action, image_dict = decode_hdf5(hdf5_file)

    if not image_dict:
        print(f"⚠️  No camera data found in {hdf5_file.name}, skipping.")
        return

    # Assume all cameras have the same number of frames
    cam0 = next(iter(image_dict.values()))
    num_frames = len(cam0)

    # Resample arrays to match the number of video frames
    qpos_rs   = resample_array(qpos,   num_frames)
    qvel_rs   = resample_array(qvel,   num_frames)
    effort_rs = resample_array(effort, num_frames)
    action_rs = resample_array(action, num_frames)

    # Save image data as mp4 videos
    for cam_name, frames in image_dict.items():
        key = f"observations/images/{cam_name}"
        stacked = np.stack(frames, axis=0)
        is_depth = False  # Assume RGB only
        suffix = "rgb_image"
        video_name = f"{cam_name}_{suffix}.rmb.mp4"
        print(f"🎞️ Saving video: {video_name}")
        save_images_to_video(stacked, rmb_dir / video_name, fps=fps, is_depth=is_depth)

    # Save non-image data as main.rmb.hdf5
    hdf5_out_path = rmb_dir / "main.rmb.hdf5"
    save_episode_hdf5(hdf5_out_path, is_sim, qpos_rs, qvel_rs, effort_rs, action_rs, fps)

    print(f"✅ Done: {episode_name}")

def process_dataset(parent_dir, out_dir, fps=30, nproc=1):
    parent_dir = Path(parent_dir)
    out_dir = Path(out_dir)

    for dataset_folder in sorted(parent_dir.iterdir()):
        print ("🔍 Checking folder: ", dataset_folder)
        if not dataset_folder.is_dir():
            continue
        print(f"\n📦 Processing folder: {dataset_folder.name}")
        hdf5_files = list(natsort.natsorted(dataset_folder.glob("episode*.hdf5")))
        args_list = [(hdf5_file, dataset_folder, out_dir, fps) for hdf5_file in hdf5_files]
        if nproc > 1:
            with Pool(nproc) as pool:
                pool.map(process_single_hdf5, args_list)
        else:
            for args in args_list:
                process_single_hdf5(args)

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 robot dataset into RGB videos.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to input dataset root folder.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output folder.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second for output video.")
    parser.add_argument('--nproc', type=int, default=1, help="Number of parallel processes.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    process_dataset(args.input_dir, args.output_dir, fps=args.fps, nproc=args.nproc)

if __name__ == "__main__":
    main()
