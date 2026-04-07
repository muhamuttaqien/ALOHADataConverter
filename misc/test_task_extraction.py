#!/usr/bin/env python3
"""
HDF5 Task Name Extractor
HDF5ファイル内のタスク情報を分析・抽出するテストツール
"""

import h5py
import argparse
from pathlib import Path
import json

def analyze_hdf5_structure(hdf5_file):
    """HDF5ファイルの構造を詳細分析"""
    print(f"🔍 Analyzing HDF5 structure: {hdf5_file}")
    print("=" * 60)
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # ファイル全体の属性
            print("📋 Root attributes:")
            for attr_name in f.attrs.keys():
                attr_value = f.attrs[attr_name]
                if isinstance(attr_value, bytes):
                    attr_value = attr_value.decode('utf-8')
                print(f"  {attr_name}: {attr_value}")
            
            if not f.attrs.keys():
                print("  (no root attributes found)")
            
            print("\n📂 Groups and Datasets:")
            
            def print_structure(name, obj):
                indent = "  " * (name.count('/'))
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/")
                    # グループの属性を表示
                    if obj.attrs.keys():
                        for attr_name in obj.attrs.keys():
                            attr_value = obj.attrs[attr_name]
                            if isinstance(attr_value, bytes):
                                attr_value = attr_value.decode('utf-8')
                            print(f"{indent}  @{attr_name}: {attr_value}")
                elif isinstance(obj, h5py.Dataset):
                    shape_str = "x".join(map(str, obj.shape))
                    print(f"{indent}📄 {name} [{shape_str}] {obj.dtype}")
                    # データセットの属性を表示
                    if obj.attrs.keys():
                        for attr_name in obj.attrs.keys():
                            attr_value = obj.attrs[attr_name]
                            if isinstance(attr_value, bytes):
                                attr_value = attr_value.decode('utf-8')
                            print(f"{indent}  @{attr_name}: {attr_value}")
            
            f.visititems(print_structure)
            
            # タスク名の候補を検索
            print("\n🎯 Task Name Detection:")
            task_candidates = []
            
            # 1. Root attributes
            for attr_name in ['task', 'task_name', 'task_description']:
                if attr_name in f.attrs:
                    value = f.attrs[attr_name]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    task_candidates.append(f"Root attribute '{attr_name}': {value}")
            
            # 2. Group attributes  
            def find_task_in_groups(name, obj):
                if isinstance(obj, h5py.Group):
                    for attr_name in ['task', 'task_name', 'name']:
                        if attr_name in obj.attrs:
                            value = obj.attrs[attr_name]
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            task_candidates.append(f"Group '{name}' attribute '{attr_name}': {value}")
            
            f.visititems(find_task_in_groups)
            
            # 3. Dataset values
            for dataset_path in ['/task', '/task_name', '/metadata/task']:
                if dataset_path in f:
                    dataset = f[dataset_path]
                    try:
                        if dataset.shape == ():  # スカラー値
                            value = dataset[()]
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            task_candidates.append(f"Dataset '{dataset_path}' (scalar): {value}")
                        elif len(dataset) > 0:  # 配列
                            value = dataset[0]
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            task_candidates.append(f"Dataset '{dataset_path}' [0]: {value}")
                    except Exception as e:
                        task_candidates.append(f"Dataset '{dataset_path}': Error reading - {e}")
            
            if task_candidates:
                print("  ✅ Found task information:")
                for candidate in task_candidates:
                    print(f"    • {candidate}")
            else:
                print("  ❌ No task information found in HDF5")
                print("  📂 Will use folder name as fallback")
            
            # 実際のタスク名抽出をテスト
            print(f"\n🎯 Extracted Task Name:")
            from convert_to_lerobot import extract_task_name_from_hdf5
            extracted_task = extract_task_name_from_hdf5(hdf5_file)
            print(f"  📝 Result: '{extracted_task}'")
            
    except Exception as e:
        print(f"❌ Error analyzing HDF5 file: {e}")

def test_multiple_files(directory):
    """ディレクトリ内の複数のHDF5ファイルを分析"""
    directory = Path(directory)
    
    hdf5_files = list(directory.rglob("episode*.hdf5"))
    
    if not hdf5_files:
        print(f"❌ No HDF5 files found in {directory}")
        return
    
    print(f"🔍 Found {len(hdf5_files)} HDF5 files")
    print("Testing task extraction from first few files...\n")
    
    task_summary = {}
    
    for i, hdf5_file in enumerate(hdf5_files[:5]):  # 最初の5個をテスト
        print(f"📄 File {i+1}: {hdf5_file.name}")
        try:
            from convert_to_lerobot import extract_task_name_from_hdf5
            task_name = extract_task_name_from_hdf5(hdf5_file)
            print(f"  📝 Task: '{task_name}'")
            
            # 親フォルダ名を記録
            folder_name = hdf5_file.parent.name
            if folder_name not in task_summary:
                task_summary[folder_name] = []
            task_summary[folder_name].append(task_name)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()
    
    print("📊 Summary by folder:")
    for folder, tasks in task_summary.items():
        unique_tasks = list(set(tasks))
        print(f"  📂 {folder}: {unique_tasks}")

def main():
    parser = argparse.ArgumentParser(description="Analyze HDF5 files for task information")
    parser.add_argument('path', help='Path to HDF5 file or directory containing HDF5 files')
    parser.add_argument('--detailed', action='store_true', help='Show detailed structure analysis')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return
    
    if path.is_file() and path.suffix == '.hdf5':
        if args.detailed:
            analyze_hdf5_structure(path)
        else:
            print(f"🔍 Testing task extraction: {path.name}")
            try:
                from convert_to_lerobot import extract_task_name_from_hdf5
                task_name = extract_task_name_from_hdf5(path)
                print(f"📝 Extracted task: '{task_name}'")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif path.is_dir():
        test_multiple_files(path)
    
    else:
        print(f"❌ Invalid path: {path}")
        print("Please provide either an HDF5 file or a directory containing HDF5 files")

if __name__ == "__main__":
    main()