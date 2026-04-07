# Lerobot Conversion

## Usage

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

| Argument | Description | Default Value |
|---|---|---|
| `--input_dir` | Path to the input HDF5 dataset directory | **Required** |
| `--output_dir` | Path to the output directory for Lerobot format | **Required** |
| `--fps` | Frames per second | `30` |
| `--task_string` | Task name or description | `"default task"` |
| `--frame_time_interval` | Time interval between frames in seconds | `0.1` |
| `--chunk_size` | Number of episodes per chunk | `3` |
| `--compressed` | Indicates that input is compressed | `False` (flag) |

## Output Layout

```text
lerobot_dataset/task_name/
├── data/
│   ├── chunk-000.parquet
│   └── chunk-001.parquet
└── meta/
    ├── info.json
    ├── episodes.jsonl
    └── tasks.jsonl
```
