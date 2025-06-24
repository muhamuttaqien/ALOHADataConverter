#!/usr/bin/env python3

import os
import h5py
import numpy as np
import argparse
import cv2
from pathlib import Path
import natsort

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

def save_episode_hdf5(hdf5_path, out_path, is_sim, qpos, qvel, effort, action, fps):
    import h5py
    with h5py.File(out_path, 'w') as f:
        f.attrs['sim'] = is_sim
        f.attrs['fps'] = fps  # fps情報を追加
        f.create_dataset('observations/qpos', data=qpos)
        f.create_dataset('observations/qvel', data=qvel)
        f.create_dataset('observations/effort', data=effort)
        f.create_dataset('action', data=action)
        # 画像データは含めない

def resample_array(arr, target_len):
    """Resample array to target length by uniform subsampling."""
    if len(arr) == target_len:
        return arr
    idxs = np.linspace(0, len(arr) - 1, target_len).astype(int)
    return arr[idxs]

def process_dataset(parent_dir, out_dir, fps=30):
    parent_dir = Path(parent_dir)
    out_dir = Path(out_dir)

    for dataset_folder in sorted(parent_dir.iterdir()):
        print ("🔍 Checking folder: ", dataset_folder)
        if not dataset_folder.is_dir():
            continue
        print(f"\n📦 Processing folder: {dataset_folder.name}")
        for i, hdf5_file in enumerate(natsort.natsorted(dataset_folder.glob("episode*.hdf5"))):
            print(f"📄 Reading file: {hdf5_file.name}")
            episode_name = f"episode_{i:06d}.rmb"
            rmb_dir = out_dir / dataset_folder.name / episode_name
            rmb_dir.mkdir(parents=True, exist_ok=True)

            is_sim, qpos, qvel, effort, action, image_dict = decode_hdf5(hdf5_file)

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
            save_episode_hdf5(hdf5_file, hdf5_out_path, is_sim, qpos_rs, qvel_rs, effort_rs, action_rs, fps)

            print(f"✅ Done: {episode_name}")

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 robot dataset into RGB videos.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to input dataset root folder.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output folder.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second for output video.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    process_dataset(args.input_dir, args.output_dir, fps=args.fps)

if __name__ == "__main__":
    main()
