# 📦ALOHA Data Converter

This repository contains scripts for converting an ALOHA HDF5 dataset into two widely used formats for robot learning:

**Lerobot format**: used in datasets hosted on Hugging Face – [Lerobot](https://huggingface.co/lerobot).

**RMB (RoboManipBaselines) format**: used in [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines), a benchmark suite for robotic manipulation.

**GR00T format**: parquet + per-camera MP4 layout compatible with [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md) training pipelines.

These scripts transform raw dataset files (typically containing robot data such as observations and actions) into efficient, structured formats compatible with their respective libraries. Both formats include metadata generation and configurable chunking of episodes.

## Features

- Converts raw HDF5 datasets to the Lerobot and RMB formats.
- Supports multiple datasets within a directory.
- Allows customization of frame time intervals, task names, and more.
- Supports input datasets that are either compressed or uncompressed.
- Outputs metadata files in JSON and JSONL formats.

⚠️ **Note:** This script is designed to work with **compressed HDF5 data**. If your dataset is in **original (raw) format**, please run the provided `compress_hdf5.py` script first to convert it before using this converter. To use the script, run the following command:

```bash
python compress_hdf5.py \
  --dataset_dir ./path/to/original_hdf5_dataset \
  --output_dir ./path/to/compressed_output \
  --nproc 4 \ 
  --quality 40 \
  --compress
```

## Requirements

Before running the conversion script, ensure you have the following Python dependencies installed:

- **numpy**: For handling numerical data and arrays.
- **pandas**: For manipulating and analyzing data, especially for DataFrame operations.
- **h5py**: For reading and writing HDF5 files.
- **pyarrow**: For working with Apache Parquet files.
- **natsort**: For naturally sorting filenames and data.
- **nopencv-python (cv2)**: For handling image data (e.g., visual observations in episodes).

You can install all the required dependencies using `pip`:

```bash
pip install numpy pandas h5py pyarrow natsort opencv-python
```

## Installation

To set up the ALOHADataConverter repository, follow these steps:

### 1. Clone the repository from GitHub:

```bash
git clone https://github.com/your-username/ALOHADataConverter.git
cd ALOHADataConverter
```

### 2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
The repository is now ready to use!

## Lerobot: Usage

The script can be executed directly from the command line. It processes the dataset and outputs the results in a custom Lerobot-compatible format.

To use the script, run the following command:

```bash
python convert_to_lerobot.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/lerobot_dataset \
  --fps 30 \
  --task_string "open drawer task" \
  --frame_time_interval 0.1 \
  --chunk_size 1000 \
  --compressed
```

## Lerobot: Command-line Arguments

The following arguments can be passed to the `convert_to_lerobot.py` script:

| Argument               | Description                                           | Default Value       |
|------------------------|-------------------------------------------------------|---------------------|
| `--input_dir`          | Path to the input HDF5 dataset directory              | **Required**        |
| `--output_dir`         | Path to the output directory for the Lerobot format   | **Required**        |
| `--fps`                | Frames per second (fps)                               | `30`                |
| `--task_string`        | Task name or description                              | `"default task"`    |
| `--frame_time_interval`| Time interval between frames in seconds               | `0.1`               |
| `--chunk_size`         | Number of episodes per chunk                          | `3`                 |
| `--compressed`         | Indicates if the output data is compressed            | `True` (flag only)  |

 ## Lerobot: Output

After the script runs, the following output will be generated in the specified `--output_dir`:

#### Data Files

Each chunk of episodes is saved in **Parquet** format. These files are named as follows:
- chunk-000.parquet
- chunk-001.parquet
- ...

After running the script, your output directory will be organized like this:

```
lerobot_dataset/task_name/
├── data/
│ ├── chunk-000.parquet
│ └── chunk-001.parquet
└── meta/
├── info.json
├── episodes.jsonl
└── tasks.jsonl
```

## RMB: Usage

The script can be executed directly from the command line. It processes the dataset and outputs the results in a custom RMB-compatible format.

To use the script, run the following command:

```bash
python convert_to_rmb.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rmb_dataset \
  --fps 30
```

## RMB: Command-line Arguments

The following arguments can be passed to the `convert_to_lerobot.py` script:

| Argument               | Description                                           | Default Value       |
|------------------------|-------------------------------------------------------|---------------------|
| `--input_dir`          | Path to the input HDF5 dataset directory              | **Required**        |
| `--output_dir`         | Path to the output directory for the RMB format       | **Required**        |
| `--fps`                | Frames per second (fps)                               | `30`                |

**Note**: There is no `--compressed` argument for this script because the output is saved as MP4 video files, which are already compressed.

 ## RMB: Output

After the script runs, the following output will be generated in the specified `--output_dir`.

#### Data Files

After running the script, your output directory will be organized like this:

```
rmb_dataset/task_name/
└── episode_000000.rmb/
    ├── cam_high_rgb_image.rmb.mp4
    ├── cam_left_wrist_rgb_image.rmb.mp4
    ├── cam_low_rgb_image.rmb.mp4
    └── cam_right_wrist_rgb_image.rmb.mp4
```

## GR00T: Usage

The script can be executed directly from the command line. It processes the dataset and outputs the results in a GR00T-compatible format (parquet + per-camera MP4s + metadata).

To use the script, run the following command:

```bash
python convert_to_gr00t.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/gr00t_dataset \
  --fps 30 \
  --cameras cam_high cam_left_wrist cam_low cam_right_wrist
```

## GR00T: Command-line Arguments

The following arguments can be passed to the `convert_to_gr00t.py` script:

| Argument               | Description                                           | Default Value       |
|------------------------|-------------------------------------------------------|---------------------|
| `--input_dir`          | Path to the input HDF5 dataset directory              | **Required**        |
| `--output_dir`         | Path to the output directory for the GR00T format     | **Required**        |
| `--fps`                | Frames per second (fps)                               | `30`                |
| `--cameras`            | Camera names to export                                | `cam_high cam_left_wrist cam_low cam_right_wrist`                |

**Note**: Each camera stream is exported as a compressed MP4 file. The vector features (`qpos`, `qvel`, `effort`, `action`) are stored in Parquet format.

## GR00T: Output

After the script runs, the following output will be generated in the specified `--output_dir`.

#### Data Files

After running the script, your output directory will be organized like this:

```
gr00t_dataset/task_name/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
├── videos/chunk-000/
│   ├── observation.images.cam_high/episode_000000.mp4
│   ├── observation.images.cam_left_wrist/episode_000000.mp4
│   ├── observation.images.cam_low/episode_000000.mp4
│   └── observation.images.cam_right_wrist/episode_000000.mp4
└── meta/
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── modality.json
    ├── info.json
    └── stats.json
```

## License

This project is not currently licensed. You are free to use the code, but please be aware that there are no official terms governing its use. If you would like to contribute or suggest a license, feel free to open an issue or pull request.
