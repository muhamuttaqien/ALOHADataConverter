# Setup and Common Notes

## Installation

Editable install:

```bash
uv pip install -e .
```

For RLDS / TFDS building, install optional dependencies too:

```bash
uv pip install -e ".[rlds]"
```

Main console scripts:

```bash
aloha-convert-rmb
aloha-convert-rlds
aloha-convert-lerobot
aloha-convert-aloha-lerobot
aloha-convert-gr00t
aloha-compress-hdf5
```

## Features

- Converts raw HDF5 datasets to Lerobot and RMB formats.
- Supports multiple datasets in a directory.
- Supports configurable frame intervals, task names, and chunk sizes.
- Supports compressed and uncompressed input datasets.
- Generates JSON / JSONL metadata files.

## Input Data Note

This converter is designed to work with compressed HDF5 data. If your data is in original (raw) format, run `compress_hdf5.py` first.

```bash
python compress_hdf5.py \
  --dataset_dir ./path/to/original_hdf5_dataset \
  --output_dir ./path/to/compressed_output \
  --nproc 4 \
  --quality 40 \
  --compress
```

## Python Requirements

- numpy
- pandas
- h5py
- pyarrow
- natsort
- opencv-python

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Setup (Alternative)

```bash
git clone https://github.com/your-username/ALOHADataConverter.git
cd ALOHADataConverter
pip install -r requirements.txt
```
