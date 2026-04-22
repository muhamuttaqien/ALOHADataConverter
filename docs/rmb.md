# RMB Conversion

## Usage

```bash
python convert_to_rmb.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rmb_dataset \
  --robot_urdf ./config/vx300s.urdf \
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
| `--robot_urdf` | URDF used for forward kinematics of EEF poses. If omitted, the converter searches `config/vx300s.urdf` and `assets/vx300s.urdf`. | `None` |

There is no `--compressed` argument because output videos are MP4-compressed by design.

`measured_eef_pose`, `command_eef_pose`, and their `_rel` variants are generated from joint states using forward kinematics on the VX300S URDF. `measured_eef_wrench` stores the raw `/observations/effort` values for the first 6 joints of each arm as-is. The gripper effort is not included because RMB expects 6 values per end effector.

## From LeRobot

You can also convert a LeRobot dataset into RMB:

```bash
python convert_to_rmb_from_lerobot.py \
  --input_dir ./path/to/lerobot_dataset \
  --output_dir ./path/to/output/rmb_dataset \
  --robot_urdf ./assets/vx300s.urdf
```

The converter first prints the LeRobot feature keys, the first Parquet schema, and the LeRobot-to-RMB mapping, then asks `y/n` before starting conversion. Use `--yes` to skip the prompt.

`convert_to_rmb_from_lerobot.py` first tries to infer the RMB mapping from LeRobot metadata:

- `meta/modality.json` for `qpos` / `qvel` / `effort` / `action` slice boundaries
- `meta/info.json` `features.*.names`
- `meta/info.json` `features.*.field_descriptions[*].indices`
- `meta/info.json` `robot_type`

If metadata is not sufficient, you can still override the layout manually. Example for a single-arm 7-DoF robot whose last joint is the gripper:

```bash
python convert_to_rmb_from_lerobot.py \
  --input_dir ./path/to/lerobot_dataset \
  --output_dir ./path/to/output/rmb_dataset \
  --robot_urdf ./assets/ws250s.urdf \
  --robot_name wx250s \
  --arm_joint_dims 7 \
  --gripper_indices 6
```

Useful options for non-ALOHA robots:

| Argument | Description | Default Value |
|---|---|---|
| `--arm_joint_dims` | Comma-separated joint dimensions for each arm in observation/action order | `7,7` |
| `--gripper_indices` | Comma-separated global gripper joint indices. Use `none` for gripperless robots | last joint of each arm |
| `--arm_fk_dims` | Comma-separated FK joint counts per arm. If omitted, gripper joints are excluded from FK by default | inferred |
| `--eef_target_links` | Comma-separated URDF target links for FK. A single value is reused for every arm | inferred from URDF |
| `--robot_name` | RMB metadata prefix for `demo_name` / `env` | `meta.info.robot_type` or `Aloha` |

When the LeRobot dataset only contains gripper joints and no arm joints, the converter still exports RMB by preserving the joint and gripper signals and zero-filling EEF pose / wrench.

To inspect whether the FK result is reasonable, you can visualize the arm geometry and EEF axes for one frame:

```bash
python misc/visualize_eef_fk.py \
  --hdf5 ./path/to/episode_0.hdf5 \
  --frame_idx 0 \
  --robot_urdf ./config/vx300s.urdf \
  --source both
```

This writes `fk_frame_0000.png` next to the input HDF5 by default. If you want an interactive window, add `--show`. In environments where the default GUI backend is unavailable, specify one explicitly, for example `--backend qt5agg`.

This utility requires `matplotlib`.

To export an MP4 over a frame range:

```bash
python misc/visualize_eef_fk.py \
  --hdf5 ./path/to/episode_0.hdf5 \
  --robot_urdf ./config/vx300s.urdf \
  --source both \
  --video_path ./fk_preview.mp4 \
  --frame_start 0 \
  --frame_stop 200 \
  --frame_stride 2 \
  --fps 15
```

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
