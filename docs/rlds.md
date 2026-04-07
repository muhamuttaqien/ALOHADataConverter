# RLDS Conversion

## Usage

Generate a TFDS / RLDS dataset package:

```bash
python convert_to_rlds.py \
  --input_dir ./path/to/hdf5_dataset \
  --output_dir ./path/to/output/rlds_dataset \
  --fps 25
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
