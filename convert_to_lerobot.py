#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import cv2
from pathlib import Path
import natsort
import argparse

# Constants
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist"]
DCAMERA_NAMES = ["dcam_high", "dcam_low"]

# Function to handle datasets inside the HDF5 file
def extract_data(f, arrays):
    for name, obj in f.items():  # Iterate over items in the root group
        if isinstance(obj, h5py.Dataset):  # If it's a dataset (not a group)
            array = obj[()]  # Extract the data from the dataset
            arrays[name] = array  # Store it in the arrays dictionary
        elif isinstance(obj, h5py.Group):  # If it's a group, recurse into it
            for sub_name, sub_obj in obj.items():
                if isinstance(sub_obj, h5py.Dataset):  # If it's a dataset inside the group
                    array = sub_obj[()]
                    arrays[f"{name}/{sub_name}"] = array  # Store the data with full path as key

# Function to update dataset keys
def update_dict_keys(arrays):
    old_keys = ['cam_high', 'cam_left_wrist', 'cam_low', 'cam_right_wrist']
    updated_arrays = {}

    for key, value in arrays.items():
        if key in old_keys:
            new_key = f'observations/images/{key}'
            updated_arrays[new_key] = value
        else:
            updated_arrays[key] = value

    return updated_arrays

# Function to load compressed HDF5 data
def decode_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()
    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]

        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            # Decode images
            emc_images = root[f'/observations/images/{cam_name}'][()]
            image_dict[cam_name] = [cv2.imdecode(img, 1) for img in emc_images]
    
    return is_sim, qpos, qvel, effort, action, image_dict

# Function to convert data and save in Lerobot format
def process_dataset(input_dir, output_dir, fps, task_string, frame_time_interval, chunk_size, compressed):
    parent_dir = Path(input_dir)

    for dataset_folder in sorted(parent_dir.iterdir()):
        if dataset_folder.is_dir():
            print(f"📦 Processing: {dataset_folder.name}")

            out_dir = Path(output_dir) / dataset_folder.name
            (out_dir / "data/chunk-000").mkdir(parents=True, exist_ok=True)
            (out_dir / "meta").mkdir(parents=True, exist_ok=True)

            episodes_meta = []
            total_frames = 0
            episode_count = 0
            frame_index = 0

            # Process each .h5 file in the dataset folder
            for hdf5_file in natsort.natsorted(dataset_folder.glob("episode*.hdf5")):

                if compressed:
                    with h5py.File(hdf5_file, 'r') as f:
                        arrays = {}
                        extract_data(f, arrays)
                        arrays = update_dict_keys(arrays)

                        obs_keys = [key for key in arrays.keys() if 'observations' in key]
                        action_keys = [key for key in arrays.keys() if 'action' in key]

                        # Process observation/action one by one to avoid memory spike
                        observation = np.concatenate([arrays[key] for key in obs_keys], axis=1).astype(np.float32)
                        action = np.concatenate([arrays[key] for key in action_keys], axis=1).astype(np.float32)

                        T = action.shape[0]
                else:
                    is_sim, qpos, qvel, effort, action, image_dict = decode_hdf5(hdf5_file)
                    T = action.shape[0]

                    arrays = {
                        "observations.qpos": qpos,
                        "observations.qvel": qvel,
                        "observations.effort": effort,
                        "action": action
                    }

                    for cam_name in CAMERA_NAMES:
                        # Only keep one episode's images in memory at a time
                        flattened_images = np.stack([img.flatten() for img in image_dict[cam_name]], axis=0)
                        arrays[f'observations.images.{cam_name}'] = flattened_images

                    obs_keys = [k for k in arrays if k.startswith('observations')]
                    act_keys = [k for k in arrays if k == 'action']

                    observation = np.concatenate([arrays[k] for k in obs_keys], axis=1).astype(np.float32)
                    action = np.concatenate([arrays[k] for k in act_keys], axis=1).astype(np.float32)

                new_data = {
                    "observation.state": observation.tolist(),
                    "action": action.tolist(),
                    "episode_index": [episode_count] * T,
                    "frame_index": list(np.arange(frame_index, frame_index + T)),
                    "timestamp": list(np.arange(T) * frame_time_interval),
                    "next.done": [False] * T,
                    "index": list(np.arange(total_frames, total_frames + T)),
                    "task_index": [0] * T
                }
                new_data["next.done"][-1] = True

                df = pd.DataFrame(new_data)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, out_dir / f"data/chunk-000/episode_{episode_count:06d}.parquet")

                # Explicitly free memory
                del observation, action, arrays, df, table, new_data
                if not compressed:
                    del image_dict
                import gc; gc.collect()

                episodes_meta.append({
                    "episode_index": episode_count,
                    "length": T,
                    "tasks": [task_string]
                })

                total_frames += T
                frame_index += T
                episode_count += 1

            # Save metadata
            obs_dim = T and observation.shape[1] or 0
            act_dim = T and action.shape[1] or 0

            features = {
                "observation.state": {"dtype": "float32", "shape": [obs_dim]},
                "action": {"dtype": "float32", "shape": [act_dim]},
                "episode_index": {"dtype": "int64", "shape": []},
                "frame_index": {"dtype": "int64", "shape": []},
                "timestamp": {"dtype": "float64", "shape": []},
                "next.done": {"dtype": "bool", "shape": []},
                "index": {"dtype": "int64", "shape": []},
                "task_index": {"dtype": "int64", "shape": []}
            }

            info = {
                "fps": fps,
                "codebase_version": "v2.1",
                "robot_type": None,
                "features": features,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None,
                "total_episodes": episode_count,
                "total_frames": total_frames,
                "chunks_size": chunk_size,
                "total_chunks": 1,
                "total_tasks": 1
            }

            # Save meta files
            with open(out_dir / "meta/info.json", "w") as f:
                json.dump(info, f, indent=2)

            with open(out_dir / "meta/episodes.jsonl", "w") as f:
                for ep in episodes_meta:
                    f.write(json.dumps(ep) + "\n")

            with open(out_dir / "meta/tasks.jsonl", "w") as f:
                f.write(json.dumps({"task_index": 0, "task": task_string}) + "\n")

            print(f"✅ Done: {dataset_folder.name} → {episode_count} episodes.\n")

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to Lerobot format")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input HDF5 dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for the Lerobot format")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second (fps)")
    parser.add_argument('--task_string', type=str, default="default task", help="Task name or description")
    parser.add_argument('--frame_time_interval', type=float, default=0.1, help="Time interval between frames (seconds)")
    parser.add_argument('--chunk_size', type=int, default=1000, help="Number of frames per chunk")
    parser.add_argument('--compressed', action='store_true', help="Indicates if the output data is compressed")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    process_dataset(args.input_dir, args.output_dir, args.fps, args.task_string, args.frame_time_interval, args.chunk_size, args.compressed)

if __name__ == "__main__":
    main()
