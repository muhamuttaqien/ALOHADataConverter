# ALOHA Data Converter

This repository contains a script for converting an HDF5 dataset into the Lerobot format. The conversion process includes transforming raw dataset files (typically containing robot data such as observations and actions) into an efficient, queryable format (Parquet) compatible with Lerobot, a robot learning dataset format.

The script provides customization options through command-line arguments, allowing you to control various aspects of the conversion, such as frame time intervals, task names, chunk sizes, and more.

## Features

- Converts raw HDF5 datasets to the Lerobot format.
- Supports multiple datasets within a directory.
- Supports controlling the number of episodes per chunk and episodes per file.
- Allows customization of frame time intervals, task names, and more.
- Outputs metadata files in JSON and JSONL formats.

## Installation

You can clone this repository or download the script directly. If you choose to clone the repository, use the following command:

```bash
git clone https://github.com/your-username/ALOH-Data-Converter.git
cd ALOHA-Data-Converter
