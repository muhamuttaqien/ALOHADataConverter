# ALOHA Data Converter

This repository contains a script for converting an ALOHA HDF5 dataset into the Lerobot format. The conversion process includes transforming raw dataset files (typically containing robot data such as observations and actions) into an efficient, queryable format (Parquet) compatible with Lerobot, a robot learning dataset format.

The script provides customization options through command-line arguments, allowing you to control various aspects of the conversion, such as frame time intervals, task names, chunk sizes, and more.

## Features

- Converts raw HDF5 datasets to the Lerobot format.
- Supports multiple datasets within a directory.
- Supports controlling the number of episodes per chunk and episodes per file.
- Allows customization of frame time intervals, task names, and more.
- Outputs metadata files in JSON and JSONL formats.

## Requirements

Before running the conversion script, ensure you have the following Python dependencies installed:

- **h5py**: For reading and writing HDF5 files.
- **numpy**: For handling numerical data and arrays.
- **pandas**: For manipulating and analyzing data, especially for DataFrame operations.
- **pyarrow**: For working with Apache Parquet files.
- **natsort**: For naturally sorting filenames and data.

You can install all the required dependencies using `pip`:

```bash
pip install h5py numpy pandas pyarrow natsort
```

## Installation

To set up the ALOHA-Data-Converter repository, follow these steps:

### 1. Clone the repository from GitHub:

```bash
git clone https://github.com/your-username/ALOHA-Data-Converter.git
cd ALOHA-Data-Converter
```

### 2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
The repository is now ready to use!

## Usage

The script can be executed directly from the command line. It processes the dataset and outputs the results in a custom Lerobot-compatible format.

To use the script, run the following command:

```bash
python convert_to_lerobot.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/lerobot_format \
  --episodes_per_file 500 \
  --chunk_size 3
```

## Command-line Arguments

The following arguments can be passed to the `convert_to_lerobot.py` script:

| Argument               | Description                                           | Default Value      |
|------------------------|-------------------------------------------------------|---------------------|
| `--input_dir`          | Path to the input HDF5 dataset directory              | **Required**        |
| `--output_dir`         | Path to the output directory for the Lerobot format   | **Required**        |
| `--episodes_per_file`  | Number of episodes per file                           | `500`               |
| `--chunk_size`         | Number of episodes per chunk                          | `3`                 |
| `--frame_time_interval`| Time interval between frames in seconds               | `0.1`               |
| `--task_string`        | Task name or description                              | `"default task"`    |
| `--fps`                | Frames per second (fps)                               | `30`                |
| `--chunk_prefix`       | Prefix for chunk filenames                            | `"chunk"`           |
| `--version`            | Print the version of the script and exit              | `None` (flag only)  |

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

## License

This project is not currently licensed. You are free to use the code, but please be aware that there are no official terms governing its use. If you would like to contribute or suggest a license, feel free to open an issue or pull request.
