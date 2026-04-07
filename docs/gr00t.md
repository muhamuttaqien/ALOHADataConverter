# GR00T Conversion

## Usage

```bash
python convert_to_gr00t.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/gr00t_dataset \
  --fps 30 \
  --cameras cam_high cam_left_wrist cam_low cam_right_wrist
```

## Command-line Arguments

| Argument | Description | Default Value |
|---|---|---|
| `--input_dir` | Path to the input HDF5 dataset directory | **Required** |
| `--output_dir` | Path to the output directory for GR00T format | **Required** |
| `--fps` | Frames per second | `30` |
| `--cameras` | Camera names to export | `cam_high cam_left_wrist cam_low cam_right_wrist` |

Each camera stream is exported as compressed MP4. Vector features (`qpos`, `qvel`, `effort`, `action`) are stored in Parquet.

## Output Layout

```text
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
