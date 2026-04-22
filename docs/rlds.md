# RLDS Conversion

## Usage

Generate a TFDS / RLDS dataset package:

```bash
python convert_to_rlds.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rlds_dataset \
  --fps 25
```

Merge multiple task folders into one RLDS package:

```bash
python convert_to_rlds.py \
  --input_dir ./path/to/hdf5_root \
  --output_dir ./path/to/output/rlds_dataset \
  --fps 25 \
  --merge_tasks \
  --merged_dataset_name aloha_multitask
```

Build TFDS artifacts if `tensorflow_datasets` is available:

```bash
python convert_to_rlds.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rlds_dataset \
  --fps 25 \
  --build_tfds \
  --overwrite
```

## Command-line Arguments

| Argument | Description | Default Value |
|---|---|---|
| `--input_dir` | Path to the input HDF5 dataset directory | **Required** |
| `--output_dir` | Path to the output directory for RLDS package | **Required** |
| `--fps` | Target FPS after resampling | `25` |
| `--val_ratio` | Fraction of episodes for validation split | `0.0` |
| `--build_tfds` | Run `tfds build` after package generation | `False` |
| `--overwrite` | Pass `--overwrite` through to `tfds build` | `False` |
| `--merge_tasks` | Merge multiple task folders under `input_dir` into one RLDS dataset package | `False` |
| `--merged_dataset_name` | Output package name used with `--merge_tasks` | Input directory name |

When `--build_tfds` is enabled, the generated builder now reads only sampled frames, streams steps instead of materializing the full episode step list, and stores RGB observations as JPEG-backed `tfds.features.Image` instead of PNG. This reduces memory pressure and improves build time.

## Output Layout

```text
rlds_dataset/task_name/
├── __init__.py
├── task_name_dataset_builder.py
├── dataset_manifest.json
├── README.md
├── CITATIONS.bib
└── setup.py
```

## Cite this references

```bibtex

```
