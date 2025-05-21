# 📦ALOHA Data Converter

This repository contains scripts for converting an ALOHA HDF5 dataset into two widely used formats for robot learning:

Lerobot format: used in datasets hosted on Hugging Face – [Lerobot](https://huggingface.co/lerobot)

RMB (RoboManipBaselines) format: used in [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines), a benchmark suite for robotic manipulation

These scripts transform raw dataset files (typically containing robot data such as observations and actions) into efficient, structured formats compatible with their respective libraries. Both formats include metadata generation and configurable chunking of episodes.

## Features

- Converts raw HDF5 datasets to the Lerobot and RMB formats.
- Supports multiple datasets within a directory.
- Allows customization of frame time intervals, task names, and more.
- Supports input datasets that are either compressed or uncompressed.
- Outputs metadata files in JSON and JSONL formats.

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

## Usage (Lerobot)

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

## Command-line Arguments

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

 ## Output

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

## Usage (RMB)

The script can be executed directly from the command line. It processes the dataset and outputs the results in a custom RMB-compatible format.

To use the script, run the following command:

```bash
python convert_to_rmb.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rmb_dataset \
  --fps 30
```

## Command-line Arguments

The following arguments can be passed to the `convert_to_lerobot.py` script:

| Argument               | Description                                           | Default Value       |
|------------------------|-------------------------------------------------------|---------------------|
| `--input_dir`          | Path to the input HDF5 dataset directory              | **Required**        |
| `--outout_dir`         | Path to the output directory for the Lerobot format   | **Required**        |
| `--fps`                | Frames per second (fps)                               | `30`                |

**Note**: There is no --compressed argument for this script because the output is saved as MP4 video files, which are already compressed.

 ## Output

After the script runs, the following output will be generated in the specified `--output_dir`:

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

## License

This project is not currently licensed. You are free to use the code, but please be aware that there are no official terms governing its use. If you would like to contribute or suggest a license, feel free to open an issue or pull request.
