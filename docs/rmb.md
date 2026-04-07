# RMB Conversion

## Usage

```bash
python convert_to_rmb.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rmb_dataset \
  --fps 25 \
  --camera_workers 4 \
  --video_preset veryfast
```

## Command-line Arguments

| Argument | Description | Default Value |
|---|---|---|
| `--input_dir` | Path to the input HDF5 dataset directory | **Required** |
| `--output_dir` | Path to the output directory for RMB format | **Required** |
| `--fps` | Frames per second | `25` |
| `--nproc` | Number of parallel episode workers | `1` |
| `--camera_workers` | Parallel camera workers per episode | `0` |
| `--video_preset` | ffmpeg/videoio preset for MP4 encoding | `veryfast` |

There is no `--compressed` argument because output videos are MP4-compressed by design.

## Output Layout

```text
rmb_dataset/task_name/
└── episode_000000.rmb/
    ├── cam_high_rgb_image.rmb.mp4
    ├── cam_left_wrist_rgb_image.rmb.mp4
    ├── cam_low_rgb_image.rmb.mp4
    ├── cam_right_wrist_rgb_image.rmb.mp4
    ├── dcam_high_rgb_image.rmb.mp4
    ├── dcam_low_rgb_image.rmb.mp4
    ├── dcam_high_depth_image.rmb.mp4
    ├── dcam_low_depth_image.rmb.mp4
    └── main.rmb.hdf5
```

## Cite this references

```bibtex
@article{RoboManipBaselines_Murooka_2025,
  title={RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulation Environments},
  author={Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi and Oh, Hanbit and Makihara, Koshi and Shirai, Keisuke and Ogata, Tetsuya and Domae, Yukiyasu},
  journal={arXiv preprint arXiv:2509.17057},
  year={2025}
}
```
