import argparse
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path
import natsort

# Function to extract data from HDF5 files
def extract_data(f, arrays):
    for name, obj in f.items():
        if isinstance(obj, h5py.Dataset):
            array = obj[()]
            arrays[name] = array
        elif isinstance(obj, h5py.Group):
            for sub_name, sub_obj in obj.items():
                if isinstance(sub_obj, h5py.Dataset):
                    array = sub_obj[()]
                    arrays[f"{name}/{sub_name}"] = array

def convert_to_lerobot(input_dir, output_dir, episodes_per_file, chunk_size, frame_time_interval, task_string, fps, chunk_prefix):
    parent_dir = Path(input_dir)
    task_index = 0
    frame_index = 0
    episode_index = 0
    total_frames = 0
    episode_count = episode_index

    # Process each dataset folder
    for dataset_folder in sorted(parent_dir.iterdir()):
        print(f"📦 Processing: {dataset_folder.name}")

        # Create necessary output directories specific to this dataset
        out_dir = Path(output_dir) / dataset_folder.name
        (out_dir / "data").mkdir(parents=True, exist_ok=True)
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)

        episodes_meta = []

        chunk_counter = 0  # For managing chunk files
        current_chunk = []

        # Process each .h5 file in the dataset folder
        for hdf5_file in natsort.natsorted(dataset_folder.glob("episode*.hdf5")):
            with h5py.File(hdf5_file, 'r') as f:
                arrays = {}
                extract_data(f, arrays)

                # Extract observation and action components
                obs_keys = [key for key in arrays.keys() if 'observations' in key]
                action_keys = [key for key in arrays.keys() if 'action' in key]

                # Stack all observation components into one array
                observation = np.concatenate([arrays[key] for key in obs_keys], axis=1).astype(np.float32)

                # Stack all action components into one array
                action = np.concatenate([arrays[key] for key in action_keys], axis=1).astype(np.float32)

                T = action.shape[0]

            # Construct the new data dict
            new_data = {
                "observation.state": observation.tolist(),
                "action": action.tolist(),
                "episode_index": [episode_count] * T,
                "frame_index": list(np.arange(frame_index, frame_index + T)),
                "timestamp": list(np.arange(T) * frame_time_interval),
                "next.done": [False] * T,
                "index": list(np.arange(total_frames, total_frames + T)),
                "task_index": [task_index] * T,
            }
            new_data["next.done"][-1] = True  # mark final frame as done

            # Save the data to current chunk
            df = pd.DataFrame(new_data)
            table = pa.Table.from_pandas(df)

            current_chunk.append(table)
            total_frames += T
            frame_index += T
            episode_count += 1

            # If the chunk size exceeds the threshold, write it to disk
            if len(current_chunk) >= chunk_size:
                pq.write_table(pa.concat_tables(current_chunk), out_dir / f"data/{chunk_prefix}-{chunk_counter:03d}.parquet")
                chunk_counter += 1
                current_chunk = []

            # Record episode metadata
            episodes_meta.append({
                "episode_index": episode_count - 1,
                "length": T,
                "tasks": [task_string],
            })

            # If the number of episodes exceeds the threshold, write episode chunk
            if episode_count % episodes_per_file == 0:
                # Write chunk to file
                pq.write_table(pa.concat_tables(current_chunk), out_dir / f"data/{chunk_prefix}-{chunk_counter:03d}.parquet")
                chunk_counter += 1
                current_chunk = []

        # Save metadata
        obs_dim = observation.shape[1]
        act_dim = action.shape[1]

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
            "data_path": f"data/{chunk_prefix}-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.parquet",
            "video_path": None,
            "total_episodes": episode_count,
            "total_frames": total_frames,
            "chunks_size": chunk_size,
            "total_chunks": chunk_counter,
            "total_tasks": 1
        }

        # Save the info JSON file
        with open(out_dir / "meta/info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Save the episodes metadata JSONL file
        with open(out_dir / "meta/episodes.jsonl", "w") as f:
            for ep in episodes_meta:
                f.write(json.dumps(ep) + "\n")

        # Save the tasks metadata JSONL file
        with open(out_dir / "meta/tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": task_index, "task": task_string}) + "\n")

        print(f"✅ Done: {dataset_folder.name} → {episode_count} episodes.\n")


def print_version():
    print("convert_to_lerobot.py - v2.1")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to Lerobot format")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input HDF5 dataset directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for the Lerobot format")
    parser.add_argument('--episodes_per_file', type=int, default=500, help="Number of episodes per file")
    parser.add_argument('--chunk_size', type=int, default=3, help="Number of episodes per chunk")
    parser.add_argument('--frame_time_interval', type=float, default=0.1, help="Time interval between frames (seconds)")
    parser.add_argument('--task_string', type=str, default="default task", help="Task name or description")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second (fps)")
    parser.add_argument('--chunk_prefix', type=str, default="chunk", help="Prefix for chunk filenames")
    parser.add_argument('--version', action='store_true', help="Print the version of the script")

    args = parser.parse_args()

    if args.version:
        print_version()
    else:
        convert_to_lerobot(
            args.input_dir,
            args.output_dir,
            args.episodes_per_file,
            args.chunk_size,
            args.frame_time_interval,
            args.task_string,
            args.fps,
            args.chunk_prefix
        )
