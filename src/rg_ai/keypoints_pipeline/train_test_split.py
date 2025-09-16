import os
import yaml
import glob
from pathlib import Path
import random
from collections import defaultdict
from typing import Dict, List
import argparse


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def extract_video_name_from_file(filename: str) -> str:
    """
    Extract the base video name from a video file.
    
    For files like: "OG20-W-C1-101-AUS-IAKOVLEVA-Lidiia-BA-Individual_seg0_Balance_Fouetté.mp4"
    Returns: "OG20-W-C1-101-AUS-IAKOVLEVA-Lidiia-BA-Individual"
    """
    basename = os.path.splitext(filename)[0]
    
    # split on _seg to get the base video name
    if '_seg' in basename:
        return basename.split('_seg')[0]
    
    return basename


def find_all_video_files(videos_folders: List[str]) -> Dict[str, List[str]]:
    """
    Find all video files in the specified folders and group them by base video name.
    
    Returns:
        Dict mapping base video name to list of file paths
    """
    video_groups = defaultdict(list)
    
    for videos_folder in videos_folders:
        if not os.path.exists(videos_folder):
            print(f"Warning: Videos folder not found: {videos_folder}")
            continue
            
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
        
        for ext in video_extensions:
            pattern = os.path.join(videos_folder, '**', ext)
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                filename = os.path.basename(file_path)
                base_video_name = extract_video_name_from_file(filename)
                video_groups[base_video_name].append(file_path)
    
    return dict(video_groups)


def split_videos_train_test(video_groups: Dict[str, List[str]], 
                           test_size: float, 
                           val_size: float,
                           random_state: int = 42) -> Dict[str, List[str]]:
    """
    Split video groups into train/test sets ensuring all clips from same video stay together.
    
    Args:
        video_groups: Dict mapping video name to list of file paths
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with 'train', 'test' keys mapping to lists of video names
    """
    random.seed(random_state)
    video_names = list(video_groups.keys())
    random.shuffle(video_names)
    
    n_videos = len(video_names)
    n_test = int(n_videos * test_size)
    n_val = int(n_videos * val_size)
    n_train = n_videos - n_test - n_val
    
    train_videos = video_names[:n_train]
    test_videos = video_names[n_train:n_train+n_test]
    val_videos = video_names[n_train+n_test:]
    
    print(f"Split summary:")
    print(f"  Total videos: {n_videos}")
    print(f"  Train: {len(train_videos)} videos ({len(train_videos)/n_videos:.2%})")
    print(f"  Test: {len(test_videos)} videos ({len(test_videos)/n_videos:.2%})")
    print(f"  Val: {len(val_videos)} videos ({len(val_videos)/n_videos:.2%})")
    
    return {
        'train': train_videos,
        'test': test_videos,
        'val': val_videos
    }


def save_split_details(split_data: Dict[str, List[str]], 
                      video_groups: Dict[str, List[str]], 
                      output_path: str):
    """
    Save the split details to a YAML file in the videos folder.
    """
    split_details = {
        'split_summary': {
            'total_videos': len(video_groups),
            'train_videos': len(split_data['train']),
            'test_videos': len(split_data['test']),
            'val_videos': len(split_data['val'])
        },
        'splits': {
            'train': split_data['train'],
            'test': split_data['test'],
            'val': split_data['val']
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(split_details, f, default_flow_style=False, indent=2)
    
    print(f"Split details saved to: {output_path}")


def create_train_test_split(config_path: str):
    """
    Main function to create train/test split based on config file.
    """
    config = load_config(config_path)
    extract_config = config.get('extract_keypoints_config', {})
    
    test_size = extract_config.get('test_size', 0.2)
    val_size = extract_config.get('val_size', 0.2)
    random_state = extract_config.get('random_state', 42)
    videos_folders = extract_config.get('videos_folders', [])
    
    if not videos_folders:
        raise ValueError("No videos_folders found in config")
    
    print(f"Creating train/test split with:")
    print(f"  Test size: {test_size}")
    print(f"  Val size: {val_size}")
    print(f"  Random state: {random_state}")
    print(f"  Videos folders: {videos_folders}")
    print()
    
    print("Finding and grouping video files...")
    video_groups = find_all_video_files(videos_folders)
    
    if not video_groups:
        raise ValueError("No video files found in specified folders")
    
    print(f"Found {len(video_groups)} unique video groups with {sum(len(files) for files in video_groups.values())} total files")
    
    print("\nCreating train/test split...")
    split_data = split_videos_train_test(video_groups, test_size, val_size, random_state)
    
    train_test_split_config = {
        'test_size': test_size,
        'val_size': val_size,
        'random_state': random_state,
        'videos_folders': videos_folders,
        'split_file_path': os.path.join(videos_folders[0], 'train_test_split.yaml')
    }
    
    config['train_test_split_config'] = train_test_split_config
    
    print(f"\nUpdating config file with train_test_split_config...")
    #save_config(config, config_path)
    
    output_folder = videos_folders[0]
    split_details_path = os.path.join(output_folder, 'train_test_split.yaml')
    
    print(f"Saving split details...")
    save_split_details(split_data, video_groups, split_details_path)
    
    return split_data, video_groups


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Create train/test split for video data')
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/rg_ai/keypoints_pipeline/config.yaml",
        help="Path to the config file",
    )
    
    args = parser.parse_args()
    
    try:
        split_data, video_groups = create_train_test_split(args.config)
        print("\nTrain/test split created successfully!")
        print("- Config file updated with train_test_split_config")
        print("- Split details saved as YAML in videos folder")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 