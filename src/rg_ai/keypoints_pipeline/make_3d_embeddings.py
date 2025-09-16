import argparse
import os
import pickle
from typing import Dict
import warnings
import glob

import cv2
import dotenv
import numpy as np
import torch
import yaml
from datetime import datetime
from tqdm import tqdm

from rg_ai.models.motionbert import MotionBERTLite, MotionBERT

from dataclasses import dataclass


@dataclass
class KeypointsInputConfig:
    base_dir: str
    categories: list[str]
    set_patterns: dict[str, str]

@dataclass
class EmbeddingsConfig:
    output_folder_name: str
    n_frames_before: int  # replaces n_frames
    n_frames_after: int # new parameter, defaults to 0 for backward compatibility
    middle_n_frames: int
    frames_step: int
    visibility_threshold: int
    model_path: str
    keypoints_input_config: KeypointsInputConfig
    do_padding: bool = False
    debug_plot_first_frame: bool = False
    deroot: bool = True
    normalize_derooting: bool = True
    do_extract_middle_n_frames: bool = False

    get_3d_keypoints: bool = False
    get_frame_keypoints: str = "last"

def get_3d_from_2d(
        frame_idx: int,  
        n_frames_before: int,
        n_frames_after: int,
        video_frames: list[dict],
        model: MotionBERTLite,
        device: torch.device,
        frame_width: int,
        frame_height: int,
        visibility_threshold: int = 0,
        person_id: int = 0, # TODO: CHECK FOR INFERENCE
        get_3d_keypoints: bool = False,
        do_padding: bool = False,
        get_frame_keypoints: str = "last",
        middle_n_frames: int = 25,
        avg_history_dim: bool = True,
        multiple_people: bool = False,
        input_keypoints_format: str = "coco17",
        do_extract_middle_n_frames: bool = False,
        ) -> np.ndarray:
    """
    Get 3D keypoints from 2D keypoints.

    Args:

        video_frames: np.array of shape 
    """
    keypoints_3d = np.array([])

    start_idx = frame_idx - n_frames_before
    end_idx = frame_idx + n_frames_after + 1  # end is exclusive
    cut_video_frames = video_frames[start_idx:end_idx]

    if multiple_people:
        keypoints_multiframes = np.array(
            [
                frame.get(person_id, {}).get("keypoints", [])
                for frame in cut_video_frames
            ]
        )
    else:
        keypoints_multiframes = cut_video_frames

    if keypoints_multiframes.size == 0:  # skip if no keypoints
        return None

    preprocessed_keypoints = model.preprocess_keypoints(
        keypoints_multiframes,
        camera=(frame_width, frame_height),
        visibility_threshold=visibility_threshold,
        input_keypoints_format=input_keypoints_format,
    )

    keypoints_tensor = torch.from_numpy(preprocessed_keypoints).float().to(device)

    with torch.inference_mode():
        if get_3d_keypoints: 
            output, keypoints_3d = model.get_3d_keypoints(keypoints_tensor, get_frame_keypoints=get_frame_keypoints)
        else:
            output = model(keypoints_tensor)

        if do_extract_middle_n_frames: # this is done so we take only the middle n_frames from the output, since we could have padded them before, also not relevant 
            middle_n_frames_half = middle_n_frames // 2

            # The middle frame in output tensor is at index n_frames_before
            middle_frame_idx_in_output = n_frames_before
            start_middle_frame_idx = middle_frame_idx_in_output - middle_n_frames_half
            end_middle_frame_idx = middle_frame_idx_in_output + middle_n_frames_half + 1

            # Ensure indices are within bounds
            start_middle_frame_idx = max(0, start_middle_frame_idx)
            end_middle_frame_idx = min(output.shape[1], end_middle_frame_idx)

            # get the middle n_frames from the output
            output = output[:, start_middle_frame_idx:end_middle_frame_idx, :, :]
        # NOTE: this can also be done in ActionHeadClassification with avg_history_dim=True, maybe better to use that

        if avg_history_dim: # for embeddings, not for 3d
            output = output.permute(
                0, 2, 3, 1
            )  # (B, T, J, C) -> (B, J, C, T), B...batch size, T...frames, J...joints, C...channels
            output = output.mean(dim=-1)

    return output, keypoints_3d

def get_representation(
    exercise_data: dict,
    exercise_name: str,
    output_dir: str,
    model: MotionBERTLite,
    device: torch.device,
    n_frames_before: int = 60,
    n_frames_after: int = 0,
    frames_step: int = 5,
    visibility_threshold: int = 0,
    do_padding: bool = False,
    get_3d_keypoints: bool = False,
    get_frame_keypoints: str = "last",
    middle_n_frames: int = 25,
    input_keypoints_format: str = "coco17",
    do_extract_middle_n_frames: bool = False,
) -> None:
    """
    Extracts and saves keypoint representations for each person in the video.

    Parameters:
    - exercise_data: Dictionary containing keypoint data for each video.
    - exercise_name: Name of the exercise.
    - output_dir: Directory where the processed keypoints will be saved.
    - model: The MotionBERTLite model used for processing keypoints.
    - device: The device (CPU or GPU) to run the model on.
    - n_frames_before: Number of frames to consider before the current frame.
    - n_frames_after: Number of frames to consider after the current frame.
    - frames_step: Step size between frames for keypoint extraction.
    - visibility_threshold: Threshold for keypoint visibility.
    - do_padding: Whether to pad the start and end of the video with the first and last frames.
    """
    for video_path in tqdm(exercise_data):
        # ---------------get resolution------------------------
        vid = cv2.VideoCapture(video_path)

        if not vid.isOpened():
            print(f"Error: Could not open video: {video_path}")
            continue

        frame_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid.release()

        people_data = {}  # store processed keypoints separately for each person
        keypoints_3d_data = {}

        video_frames = exercise_data[video_path]
        
        # TODO: comment this out when testing with just longer extracted videos
        if do_padding:
            first_frame = video_frames[0]
            last_frame = video_frames[-1]
            padding_start = [first_frame] * n_frames_before
            padding_end = [last_frame] * n_frames_after
            video_frames = padding_start + video_frames + padding_end
            total_frames += n_frames_before + n_frames_after + 1 # +1 for the middle frame
        
        else:
            total_frames = len(video_frames) # + 1 #??

        # GET KEYPOINTS AND EMBEDDINGS
        #  each frame's keypoints
        for frame_idx in range(
            n_frames_before,  # Start after we have enough past frames
            total_frames - n_frames_after,  # End before we run out of future frames
            frames_step,
        ):
            for person_id in video_frames[frame_idx]:
                output, keypoints_3d = get_3d_from_2d(
                    frame_idx=frame_idx,
                    n_frames_before=n_frames_before,
                    n_frames_after=n_frames_after,
                    video_frames=video_frames,
                    model=model,
                    device=device,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    visibility_threshold=visibility_threshold,
                    person_id=person_id,
                    get_3d_keypoints=get_3d_keypoints,
                    do_padding=do_padding,
                    get_frame_keypoints=get_frame_keypoints,
                    avg_history_dim=True, # HACK - we hardcode it here, but we could also do this parameter in actionheadclassifiation
                    multiple_people=True, # HACK: we have different input to here than in inference
                    middle_n_frames=middle_n_frames,
                    input_keypoints_format=input_keypoints_format,
                    do_extract_middle_n_frames=do_extract_middle_n_frames,
                )

                if person_id not in people_data:
                    people_data[person_id] = []
                    keypoints_3d_data[person_id] = [] if get_3d_keypoints else None

                output_np = output.cpu().detach().numpy()

                if np.isnan(output_np).any():
                    raise ValueError(f"NaN values in output for person {person_id} at frame {frame_idx}")

                people_data[person_id].append(output_np)
                if get_3d_keypoints:
                    keypoints_3d_data[person_id].append(keypoints_3d)
                

        # output for each person in a separate file
        for person_id, processed_keypoints in people_data.items():
            video_output_path = os.path.join(
                output_dir,
                f"{exercise_name}_{os.path.splitext(os.path.basename(video_path))[0]}_person_{person_id}.pkl",
            )
            if get_3d_keypoints:
                output_dir_3d = os.path.join(os.path.dirname(output_dir), "3d_keypoints")
                os.makedirs(output_dir_3d, exist_ok=True)
                video_output_path_3d = os.path.join(
                    output_dir_3d,
                    f"{exercise_name}_{os.path.splitext(os.path.basename(video_path))[0]}_person_{person_id}_3d.pkl",
                )
                with open(video_output_path_3d, "wb") as f:
                    pickle.dump(keypoints_3d_data[person_id], f)

            with open(video_output_path, "wb") as f:
                pickle.dump(processed_keypoints, f)

        # clear people_data to free up memory
        people_data.clear()

def save_set(
    exercise_data: dict,
    exercise_name: str,
    set_name: str,
    embeddings_config: EmbeddingsConfig,
    input_keypoints_format: str = "coco17", # or blazepose33
) -> None:
    """
    Save the set of keypoints for a given set name.

    Parameters:
    - set_name: Name of the set to save the keypoints for.
    - exercise_data: Dictionary containing keypoint data for each video.
    - embeddings_config: Configuration for the embeddings generation.
    """
    output_dir = os.path.join(os.getenv("EMBEDDINGS_3D_SAVE_FOLDER"), embeddings_config.output_folder_name, set_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionBERTLite(lite=False, model_ckpt=embeddings_config.model_path, device=device, deroot=embeddings_config.deroot, normalize_derooting=embeddings_config.normalize_derooting, input_keypoints_format=input_keypoints_format)
    #model = MotionBERT(model_ckpt=embeddings_config.model_path, device=device)

    get_representation(
        exercise_data=exercise_data,
        exercise_name=exercise_name,
        output_dir=output_dir,
        model=model,
        device=device,
        n_frames_before=embeddings_config.n_frames_before,
        n_frames_after=embeddings_config.n_frames_after,
        frames_step=embeddings_config.frames_step,
        visibility_threshold=embeddings_config.visibility_threshold,
        do_padding=embeddings_config.do_padding,
        get_3d_keypoints=embeddings_config.get_3d_keypoints,
        get_frame_keypoints=embeddings_config.get_frame_keypoints,
        middle_n_frames=embeddings_config.middle_n_frames,
        input_keypoints_format=input_keypoints_format,
        do_extract_middle_n_frames=embeddings_config.do_extract_middle_n_frames,
    )

def load_problematic_videos(experiment_path: str) -> set:
    """Load list of problematic videos to skip processing."""
    problematic_videos = set()
    problematic_files = glob.glob(os.path.join(experiment_path, "**/problematic_videos*.pkl"), recursive=True)
    for file_path in problematic_files:
        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    problematic_videos.update([item[0] for item in data])

        except Exception as e:
            warnings.warn(f"Could not load problematic videos from {file_path}: {e}")
    return problematic_videos

if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/rg_ai/keypoints_pipeline/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--keypoints_dir",
        type=str,
        help="Directory containing the keypoints data. If provided, overrides the base_dir in config.",
    )
    args = parser.parse_args()

    # ------------------------ Config and output folder ------------------------
    with open(args.config, 'r') as file:
        yaml_embeddings_config = yaml.safe_load(file).get("embeddings_config")

    embeddings_config = EmbeddingsConfig(**yaml_embeddings_config)
    keypoints_input_config = KeypointsInputConfig(**embeddings_config.keypoints_input_config)
    
    if args.keypoints_dir:
        keypoints_input_config.base_dir = args.keypoints_dir
        print(f"Using provided keypoints directory: {args.keypoints_dir}")
    
    if not embeddings_config.output_folder_name:
        embeddings_config.output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Config 'output_folder_name' is empty, using generated name: {embeddings_config.output_folder_name}")
    
    output_dir = os.path.join(os.getenv("EMBEDDINGS_3D_SAVE_FOLDER"), embeddings_config.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"EMBEDDINGS_OUTPUT_DIR={output_dir}\n")

    # ----------------- Append previous config to the new one -------------
    previous_config = os.path.join(keypoints_input_config.base_dir, "config.yaml")
    if os.path.exists(previous_config):
        with open(previous_config, 'r') as file:
            previous_config = yaml.safe_load(file)
        
        # Check if keypoints extraction used train_test_split = False
        if not previous_config.get("train_test_split", True):
            print("Keypoints were extracted without train/test split. Using 'all' set pattern.")
            keypoints_input_config.set_patterns = {"all": "all_set_*.pkl"}
        
        model_name = previous_config.get("model", {}).get("name")
        if model_name == "BlazePose":
            input_keypoints_format = "blazepose33"
        else:
            input_keypoints_format = "coco17"
            
        new_config = {
            "extract_keypoints_config": previous_config,
            "embeddings_config": yaml_embeddings_config,
        }
        with open(os.path.join(output_dir, "config.yaml"), 'w') as file:
            yaml.dump(new_config, file)

    # ------------------------ Make embeddings ------------------------
    for set_name, pattern in keypoints_input_config.set_patterns.items():
        input_pkl_files = glob.glob(os.path.join(keypoints_input_config.base_dir, pattern))
        
        current_data = {}
        for input_path in input_pkl_files:
            with open(input_path, "rb") as f:
                data = pickle.load(f)
                current_data.update(data) 
            print(f"Loaded data from {input_path}")

        problematic_videos = load_problematic_videos(keypoints_input_config.base_dir)
        if problematic_videos:
            print(f"Found {len(problematic_videos)} problematic videos to skip")

        if problematic_videos:
            current_data = {
                video_path: video_data 
                for video_path, video_data in current_data.items()
                if os.path.basename(video_path) not in problematic_videos
            }

        for category_name in current_data.keys():#keypoints_input_config.categories:
            print(f"Processing category: {category_name}")

            if not input_pkl_files:
                print(f"  No {set_name} files found matching pattern '{pattern}' in {keypoints_input_config.base_dir}")
                continue

            save_set(
                exercise_data=current_data[category_name],
                exercise_name=category_name,
                set_name=set_name,
                embeddings_config=embeddings_config,
                input_keypoints_format=input_keypoints_format,
            )
    print(f"Embeddings saved to {output_dir}")
    print("Done.")

