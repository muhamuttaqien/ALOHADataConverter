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
  --robot_urdf ./assets/vx300s.urdf \
  --mapping_config ./config/lerobot_mapping.example.json
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
| `--mapping_config` | JSON file for explicit LeRobot-to-RMB remapping. Field names are resolved against the configured LeRobot feature sources before modality slicing | `None` |
| `--skip_static_eef` | Skip episodes whose configured EEF pose/relative sources are static, and remove any existing output for those episodes | `False` |

When the LeRobot dataset only contains gripper joints and no arm joints, the converter still exports RMB by preserving the joint and gripper signals and zero-filling EEF pose / wrench.

If a LeRobot dataset has lost its `meta/*.json/jsonl` files but still has Parquet episodes and videos, rebuild the minimum converter-readable metadata from a valid dataset with the same feature schema:

```bash
python misc/rebuild_lerobot_meta.py \
  --dataset_dir ./data/lerobot_yubi/lerobot_v21_ph2_success_tape \
  --template_dataset_dir ./data/lerobot_yubi/lerobot_v21_ph2_success_handover \
  --task task \
  --with_stats
```

The script rewrites `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, and `meta/episodes_stats.jsonl`. It refuses to overwrite non-empty metadata unless `--force` is specified. The template dataset must match the Parquet feature schema; the script copies `features` from the template and recalculates episode count, frame count, task count, video count, chunks, and split ranges from the target dataset.

`--mapping_config` is useful when `action` contains hand roots, controller poses, or mixed signals rather than pure joint targets. It can also read non-standard LeRobot feature columns such as `action.left_controller.relative` or `observation.pose.left_hand_root.absolute`. The file can override only the ambiguous RMB targets and leave the rest on automatic inference.

Rebuilding metadata does not recover missing robot motion. For example, the inspected `lerobot_v21_ph2_success_tape` Parquet files still contain static hand-root and controller columns, so use `--skip_static_eef` during RMB conversion if those static EEF episodes should be rejected.

If you do not want raw `observation.state` / `action` to be copied into `measured_joint_pos` / `command_joint_pos`, add `default_targets` and set them to `empty` or `remaining`.

Supported target keys in `rmb_mappings`:

- `measured_joint_pos`
- `measured_joint_pos_rel`
- `measured_joint_vel`
- `command_joint_pos`
- `command_joint_pos_rel`
- `measured_gripper_joint_pos`
- `measured_gripper_joint_pos_rel`
- `command_gripper_joint_pos`
- `command_gripper_joint_pos_rel`
- `measured_eef_pose`
- `command_eef_pose`
- `measured_eef_pose_rel`
- `command_eef_pose_rel`
- `measured_eef_wrench`

Each target can use one of these selectors:

- `fields`: explicit LeRobot field names from `meta/info.json`
- `indices`: explicit source indices
- `slice`: `[start, end]`

`source` defaults to the target-level `source` when a target uses `groups`. Otherwise set `source` directly on each target or group. The source must be a feature key present in `meta/info.json` and in the Parquet schema.

Supported `default_targets` policies:

- `auto`: keep the current automatic fallback
- `remaining`: keep only source dims not already consumed by configured RMB targets
- `empty`: write a zero-width dataset for that RMB target

`measured_eef_pose`, `command_eef_pose`, `measured_eef_pose_rel`, `command_eef_pose_rel`, and `measured_eef_wrench` may use `groups` to describe one source block per end effector. Absolute pose groups must be 7 dims each: `x y z qx qy qz qw`. Relative EEF pose groups may be 6D RMB deltas or 7D relative poses; 7D relative poses are converted to 6D `dx dy dz rx ry rz` deltas. Wrench groups may use up to 6 dims and are zero-padded to 6 when shorter.

EEF pose and relative-pose groups can also apply a signed axis transform before writing RMB:

```json
{
  "eef_transform": {
    "axes": ["x", "-z", "y"]
  }
}
```

This means output `x = input x`, output `y = -input z`, and output `z = input y`. For absolute 7D poses, both translation and quaternion rotation matrices are transformed. For relative 6D/7D deltas, both `dx dy dz` and `rx ry rz` are transformed. Put `eef_transform` at the top level to apply it to all EEF targets, or put `transform` on an EEF target/group for a narrower override. Use this only after checking the source coordinate convention; an incorrect axis transform can make the EEF move in a plausible-looking but wrong direction.

For quick trials without editing JSON, pass the same correction on the command line:

```bash
python convert_to_rmb_from_lerobot.py \
  --input_dir ./data/lerobot_yubi/lerobot_v21_ph2_success_handover \
  --output_dir ./data/rmb_yubi_trial \
  --mapping_config ./config/lerobot_mapping.genoma-v2_real.json \
  --eef_transform_axes x,-z,y \
  --yes
```

When the first axis is negative, either `--eef_transform_axes -x,y,z` or the shell-safe `--eef_transform_axes=-x,y,z` form can be used. `--eef_transform_axes` overrides top-level `eef_transform` in the mapping file. Per-group `transform` entries still take precedence.

Example:

```json
{
  "default_targets": {
    "measured_joint_pos": "empty",
    "command_joint_pos": "empty"
  },
  "rmb_mappings": {
    "measured_gripper_joint_pos": {
      "source": "observation.state",
      "fields": ["left_right_finger_joint", "right_right_finger_joint"]
    },
    "command_gripper_joint_pos": {
      "source": "action",
      "fields": ["left_right_finger_joint", "right_right_finger_joint"]
    },
    "measured_eef_pose": {
      "source": "observation.state",
      "groups": [
        {
          "fields": [
            "left_hand_root/x",
            "left_hand_root/y",
            "left_hand_root/z",
            "left_hand_root/qx",
            "left_hand_root/qy",
            "left_hand_root/qz",
            "left_hand_root/qw"
          ]
        },
        {
          "fields": [
            "right_hand_root/x",
            "right_hand_root/y",
            "right_hand_root/z",
            "right_hand_root/qx",
            "right_hand_root/qy",
            "right_hand_root/qz",
            "right_hand_root/qw"
          ]
        }
      ]
    },
    "command_eef_pose": {
      "source": "action",
      "groups": [
        {
          "fields": [
            "left_hand_root/x",
            "left_hand_root/y",
            "left_hand_root/z",
            "left_hand_root/qx",
            "left_hand_root/qy",
            "left_hand_root/qz",
            "left_hand_root/qw"
          ]
        },
        {
          "fields": [
            "right_hand_root/x",
            "right_hand_root/y",
            "right_hand_root/z",
            "right_hand_root/qx",
            "right_hand_root/qy",
            "right_hand_root/qz",
            "right_hand_root/qw"
          ]
        }
      ]
    }
  }
}
```

The same example is included as [`config/lerobot_mapping.example.json`](/home/motoda/src/ALOHADataConverter/config/lerobot_mapping.example.json).

### Genoma v2 real dataset mapping

The `lerobot_v21_ph2_success_handover` dataset stores controller, hand, and gripper signals as separate LeRobot features, not as a single `action` vector:

- `observation.state`: footpedal buttons only
- `action.left_controller.relative`: left controller command delta, 7 dims
- `action.right_controller.relative`: right controller command delta, 7 dims
- `observation.pose.left_hand_root.absolute`: left measured hand-root pose, 7 dims
- `observation.pose.right_hand_root.absolute`: right measured hand-root pose, 7 dims
- `observation.state.left_gripper`, `observation.state.right_gripper`: measured gripper positions
- `action.left_gripper.absolute`, `action.right_gripper.absolute`: command gripper positions
- `action.left_gripper.relative`, `action.right_gripper.relative`: command gripper deltas
- `observation.image.left`, `observation.image.right`, `observation.image.center`: RGB video streams

For this dataset, use [`config/lerobot_mapping.genoma-v2_real.json`](/home/motoda/src/ALOHADataConverter/config/lerobot_mapping.genoma-v2_real.json). The mapping treats the dataset as an EEF-pose dataset with gripper signals but no exported arm joints.

The `action.left_controller.relative` and `action.right_controller.relative` names are easy to confuse with RMB's absolute pose datasets. In the inspected `lerobot_v21_ph2_success_handover` episodes, `action.*_controller.relative[i]` is the 7D relative transform from `observation.pose.*_hand_root.absolute[i]` to frame `i + 1`. Store it as `command_eef_pose_rel`; the converter integrates it with `measured_eef_pose` to produce `command_eef_pose`.

Some Genoma v2 exports do not contain hand/controller motion even though HMD pose and gripper signals change. In the inspected `lerobot_v21_ph2_success_right_sps` and `lerobot_v21_ph2_success_tape` folders, `observation.pose.left_hand_root.absolute`, `observation.pose.right_hand_root.absolute`, `action.left_controller.relative`, and `action.right_controller.relative` are static for every episode. For those folders, all EEF `_rel` datasets are expected to be zero unless the original LeRobot export is regenerated with hand/controller pose motion. Mapping HMD pose into EEF pose would create nonzero values, but it would not be an end-effector signal.

When a configured `_rel` source column is absent, the converter falls back to absolute pose data: `measured_eef_pose_rel` is computed from `measured_eef_pose`, and `command_eef_pose_rel` is computed from `command_eef_pose`. If no separate command absolute pose is available, command pose mirrors measured pose for this fallback. The fallback cannot recover motion when the absolute hand/controller pose columns are themselves static.

Use `--skip_static_eef` to reject those static EEF episodes during conversion. The converter prints a `Static EEF input` error for each rejected episode and removes that episode's output directory if it already exists.

| RMB target | LeRobot source |
|---|---|
| `measured_eef_pose` | `observation.pose.left_hand_root.absolute` + `observation.pose.right_hand_root.absolute` |
| `command_eef_pose` | integrated from `measured_eef_pose` + `command_eef_pose_rel` |
| `command_eef_pose_rel` | `action.left_controller.relative` + `action.right_controller.relative` |
| `measured_joint_pos` | empty |
| `command_joint_pos` | empty |
| `measured_gripper_joint_pos` | `observation.state.left_gripper` + `observation.state.right_gripper` |
| `command_gripper_joint_pos` | `action.left_gripper.absolute` + `action.right_gripper.absolute` |
| `command_gripper_joint_pos_rel` | `action.left_gripper.relative` + `action.right_gripper.relative` |
| `measured_eef_wrench` | zero-filled |

Run:

```bash
python convert_to_rmb_from_lerobot.py \
  --input_dir ./data/lerobot_yubi/lerobot_v21_ph2_success_handover \
  --output_dir ./data/rmb_yubi \
  --mapping_config ./config/lerobot_mapping.genoma-v2_real.json \
  --yes
```

Expected output datasets in `main.rmb.hdf5` include:

- `measured_eef_pose`: `(T, 14)` for left and right `x y z qx qy qz qw`
- `command_eef_pose`: `(T, 14)` for integrated left and right command pose `x y z qx qy qz qw`
- `measured_eef_pose_rel` / `command_eef_pose_rel`: `(T, 12)` for left and right `dx dy dz rx ry rz`
- joint datasets: `(T, 0)`
- measured and command gripper datasets: `(T, 2)`
- `left_rgb_image.rmb.mp4`, `right_rgb_image.rmb.mp4`; `center_rgb_image.rmb.mp4` is exported only when the source video exists

To validate the converted RMB EEF trajectories, render a preview MP4:

```bash
python misc/visualize_rmb_dataset.py \
  --input_dir ./data/rmb_yubi \
  --output_path ./data/rmb_yubi/rmb_yubi_eef_preview.mp4 \
  --robot_urdf ./assets/vx300s.urdf \
  --eef_mode relative \
  --relative_origin urdf_zero \
  --ik_preview \
  --source both \
  --arm both \
  --max_episodes 1 \
  --frame_stride 4 \
  --max_frames_per_episode 300
```

The tool searches recursively for `main.rmb.hdf5`, so `--input_dir` can point to an RMB root, one dataset directory, one `episode_*.rmb` directory, or a `main.rmb.hdf5` file. With `--eef_mode relative`, it loads the ALOHA/VX300S URDF, starts from the all-zero FK EEF pose, integrates `measured_eef_pose_rel` and `command_eef_pose_rel`, and draws the resulting left/right EEF trajectories with a faint URDF reference skeleton. `--ik_preview` then tries to reconstruct a virtual joint trajectory for the rendered EEF targets using damped least-squares IK with URDF joint limits. The script prints median/max position and rotation residuals, so large residuals should be treated as unreachable targets or IK failure rather than recovered robot motion. Use `--eef_mode absolute` to draw stored `measured_eef_pose` and `command_eef_pose` directly. The tool also overlays available episode camera videos and writes a single MP4. It requires `matplotlib` and `opencv-python`. For `rmb_yubi`, original joint datasets are expected to be zero-width; the joint panels show IK-preview joints when `--ik_preview` is enabled.

For faster iteration, keep the default geometric IK Jacobian and reduce rendering cost:

```bash
python misc/visualize_rmb_dataset.py \
  --input_dir ./data/rmb_yubi/lerobot_v21_ph2_success_handover/episode_000000.rmb \
  --output_path /tmp/rmb_yubi_fast_ik_preview.mp4 \
  --robot_urdf ./assets/vx300s.urdf \
  --eef_mode relative \
  --ik_preview \
  --source measured \
  --frame_stride 20 \
  --max_frames_per_episode 80 \
  --no_camera \
  --figure_scale 0.55
```

`--ik_jacobian geometric` is the default and avoids finite-difference FK calls per joint. Use `--ik_jacobian numerical` only for debugging the Jacobian.

If the EEF appears to move in the wrong direction, inspect the raw LeRobot controller-relative convention before changing the converter:

```bash
python misc/tune_lerobot_eef_relative.py \
  --dataset_dir ./data/lerobot_yubi/lerobot_v21_ph2_success_handover \
  --episode_index 0 \
  --frame_stride 10 \
  --max_frames 80 \
  --top_k 12 \
  --output_path /tmp/eef_relative_candidate.mp4 \
  --render_top 3
```

The tool compares `observation.pose.*_hand_root.absolute[i + 1]` against a one-step prediction from `observation.pose.*_hand_root.absolute[i]` and `action.*_controller.relative[i]` under several axis/sign/frame hypotheses. The printed `axes=...` value can be passed directly to `--eef_transform_axes` when it represents the desired LeRobot-to-RMB basis change. For the inspected handover episode, the best one-step match is the current LeRobot convention, `axes=x,y,z ... trans=world rot=right`, with near-zero one-step error. If visualization still disagrees with ALOHA, the likely fix is a LeRobot-world to ALOHA/RMB-world axis transform on the absolute and relative EEF groups, not a change to how `action.*_controller.relative` is integrated within LeRobot coordinates.

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
