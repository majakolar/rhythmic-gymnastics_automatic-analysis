import argparse
import os
import pickle
from datetime import datetime
import time
import shutil
import cv2
import dotenv
import numpy as np
import pandas as pd
import yaml
import warnings
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

from rg_ai.models.keypoint_models import MODEL_CLASSES, RTMO, YOLO11
from rg_ai.utils.data_augmentation import DataAugmentor
from rg_ai.utils.utils_extract_keypoints import \
    extract_video_paths, \
    get_exercise_name, \
    extract_people_per_video, \
    filter_first_frame, \
    filter_next_frame, \
    add_frames_for_3d, \
    remove_problematic_videos, \
    interpolate_keypoints
from rg_ai.utils.plotting import plot_keypoints_on_frame_opencv

from dataclasses import dataclass 

@dataclass
class ExtractKeypointsConfig:
    videos_folders: list[str]
    keypoints_experiment_name: str
    filtered_subdirs: list[str]
    base_save_dir: str
    annotated_videos_dir: str
    set_names: list[str]
    test_size: float
    val_size: float
    random_state: int

    add_N_frames_before: int
    random_scaling_factor_range: list[float]
    same_person_threshold: int
    model: dict
    currently_plotting: bool
    filter_to_one_skeleton: str
    max_interpolation_gap: int
    force_recompute: bool = False

    train_test_split: bool = True

def group_videos_by_exercise(video_paths, master_folder):
    """Group videos by exercise to process them exercise by exercise.
    The output is a nested dictionary of format: {exercise_name: {video_path: []}}
    where the empty list is a placeholder for keypoint data.
    """
    videos_by_exercise = {}
    for vp_tuple in video_paths: # vp_tuple is (video_path, folder_name)
        video_path_item, _ = vp_tuple 
        ex_name = get_exercise_name(video_path_item, master_folder)
        if ex_name not in videos_by_exercise:
            videos_by_exercise[ex_name] = {}
        videos_by_exercise[ex_name][video_path_item] = [] 
    return videos_by_exercise


def main(
        experiment_base_save_dir: str,
        config: ExtractKeypointsConfig,
        set_name: str, 
        videos_folder: str, 
        keypoints_filedate: str,
        master_folder: str
    ):
    """
    Main function to extract keypoints from videos.
    Args:
        experiment_base_save_dir(str): path to the base save directory
        config(ExtractKeypointsConfig): configuration object
        set_name(str): name of the set to extract keypoints from
        videos_folder(str): path to the videos folder
        keypoints_filedate(str): date of the keypoints file, used for naming the keypoints file, experiment identifier
        master_folder(str): name of the master folder, which is the parent folder of the other videos/video folders/video profiles
    """
    np.random.seed(config.random_state)

    # ------------------- Setup paths -------------------------
    videos_type = videos_folder.split("/")[-1] # e.g. "processed_videos"

    experiment_annotated_videos_dir = os.path.join(config.annotated_videos_dir, keypoints_filedate)
    os.makedirs(experiment_annotated_videos_dir, exist_ok=True)

    if config.currently_plotting:
        # save the config in the folder
        with open(os.path.join(experiment_annotated_videos_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    incomplete_log_filename_stem = f"incomplete_videos_{videos_type}"
    problematic_log_filename_stem = f"problematic_videos_{videos_type}"
    missing_keypoints_frames_log_filename_stem = f"missing_keypoints_frames_{videos_type}"
    
    incomplete_videos_pickle_path = os.path.join(experiment_base_save_dir, f"{incomplete_log_filename_stem}_{set_name}.pkl")
    incomplete_videos_txt_path = os.path.join(experiment_base_save_dir, f"{incomplete_log_filename_stem}.txt")
    problematic_videos_pickle_path = os.path.join(experiment_base_save_dir, f"{problematic_log_filename_stem}_{set_name}.pkl")
    problematic_videos_txt_path = os.path.join(experiment_base_save_dir, f"{problematic_log_filename_stem}.txt")
    missing_keypoints_frames_pickle_path = os.path.join(experiment_base_save_dir, f"{missing_keypoints_frames_log_filename_stem}_{set_name}.pkl")
    missing_keypoints_frames_txt_path = os.path.join(experiment_base_save_dir, f"{missing_keypoints_frames_log_filename_stem}.txt")

    keypoints_pickle_filename = f"{set_name}_set_{videos_type}.pkl"
    pickle_save_path = os.path.join(experiment_base_save_dir, keypoints_pickle_filename)


    # Keeping track of problematic videos across all exercises
    all_exercises_data = {}
    total_problematic_videos = []
    total_problematic_videos_count = 0
    total_incomplete_annotated_videos = []
    total_incomplete_annotated_videos_count = 0
    video_paths_original = extract_video_paths(videos_folder, master_folder, config.filtered_subdirs)

    if len(video_paths_original) == 0:
        raise ValueError(f"No video paths found for {videos_folder}")

    
    # TODO: make this cleaner?
    if config.train_test_split:
        # ------------------- Train Val Test split -------------------------
        # NOTE: needed split, important since sometimes we need to generate just a specific set
        # TODO: put this outside the function, or give the option to generate "all" sets
        train_val_paths, test_paths = train_test_split(video_paths_original, test_size=config.test_size, random_state=config.random_state)
        train_paths, val_paths = train_test_split(train_val_paths, test_size=config.val_size, random_state=config.random_state)
        
        if set_name == "train":
            current_set_video_paths = train_paths
            data_augmentation = False
            normalize_skeleton = False
        elif set_name == "val":
            current_set_video_paths = val_paths
            data_augmentation = False
            normalize_skeleton = False
        elif set_name == "test":
            current_set_video_paths = test_paths
            data_augmentation = False
            normalize_skeleton = False

    else:
        current_set_video_paths = video_paths_original
        data_augmentation = False
        normalize_skeleton = False

    print(f"Number of {set_name} videos:", len(current_set_video_paths))
    if not current_set_video_paths:
        print(f"No videos to process for {set_name} set.")
        return None # TODO: handle this better

    # ------------------------- Model ---------------------------------

    model_class = MODEL_CLASSES[config.model["name"]]

    if config.model["name"] == "BlazePose":
        # BlazePose uses different parameters
        model2d = model_class(
            model_ckpt=config.model.get("ckpt", None),  # Not used but kept for compatibility
            input_size=config.model.get("input_size", [640, 640]),
            min_detection_confidence=config.model.get("min_detection_confidence", 0.7),
            min_tracking_confidence=config.model.get("min_tracking_confidence", 0.7),
            static_image_mode=config.model.get("static_image_mode", True),
        )
    else:
        # For RTMO and YOLO11 models
        model2d = model_class(
            model_ckpt=config.model["ckpt"],
            input_size=config.model["input_size"], 
            provider_priority=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    # -----------------------------------------------------------------

    videos_by_exercise = group_videos_by_exercise(current_set_video_paths, master_folder)
    videos_missing_keypoints = {}
    for exercise_name, video_paths in videos_by_exercise.items():
        print(f"Processing exercise: {exercise_name}")
        print(f"  Number of videos: {len(video_paths)}")
        
    # 1. PER EXERCISE LOOP
    for exercise_name, videos_for_exercise in tqdm(videos_by_exercise.items(), desc=f"Processing exercises for {set_name} set", unit="exercise"):

        # Data specific to the current exercise
        videos_missing_keypoints[exercise_name] = {} # format: {video_path: missing_frames:[frames_indices], total_frames: total_frames}

        # Keeping track of problematic videos across current exercise
        exercise_incomplete_videos = []
        exercise_problematic_videos = []
        exercise_problematic_videos_count = 0

        # 2. PER VIDEO LOOP
        for video_path in tqdm(videos_for_exercise, desc=f"Processing videos for {exercise_name}", unit="video", leave=False):
            # Init some variables per video, regarding the last frame with detected keypoints, before processing frames
            last_valid_frame_people_dict = {}
            last_valid_frame_idx = -1
            _people_dict_from_last_filter_success = {} 
            _mean_kps_from_last_filter_success = np.array([])    
            current_frame = -1
            
            # Factors and augmentation init
            random_scaling_factor = np.random.uniform(config.random_scaling_factor_range[0], config.random_scaling_factor_range[1])
            same_person_threshold = config.same_person_threshold
            augmentor = DataAugmentor()

            video_name = video_path.split("/")[-1].split(".")[:-1]
            video_name = ".".join(video_name)

            vid = cv2.VideoCapture(video_path)
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            frame_shape = (height, width)

            if not vid.isOpened():
                print(f"Error: Could not open video: {video_path}")
                # TODO: maybe remove exercise_problematic_videos?
                exercise_problematic_videos.append([video_path, -1, "Could not open video"])
                exercise_problematic_videos_count +=1
                total_problematic_videos.append([video_path, -1, "Could not open video"])
                total_problematic_videos_count += 1
                continue

            total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            videos_missing_keypoints[exercise_name][video_path] = {"missing_frames": list(), "total_frames": total_frames}

            if config.currently_plotting:
                # Annotated videos for THIS exercise, within THIS experiment
                annotated_exercise_dir = os.path.join(experiment_annotated_videos_dir, exercise_name)
                os.makedirs(annotated_exercise_dir, exist_ok=True)
                annotated_video_path = os.path.join(annotated_exercise_dir, f"{video_name}_annotated_{config.model['name']}.mp4")

                if not config.force_recompute and os.path.exists(annotated_video_path):
                    print(f"\nSkipping already processed annotated video: {video_path.split('/')[-1].split('.')[0]}")
                    continue  # TODO: handle this better, this is inside the currently_plotting statement

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

            current_video_frames_data = [] 

            # 3. PER FRAME LOOP
            with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frame", leave=False) as pbar:
                while True:
                    pbar.update(1)
                    ret, frame = vid.read()
                    current_frame += 1
                    if not ret:
                        break

                    if not np.any(frame):
                        print("Black frame detected, skipping...")
                        if current_frame == 0:
                            exercise_problematic_videos_count += 1
                            exercise_problematic_videos.append([video_path, current_frame, "First frame black"])
                            total_problematic_videos.append([video_path, current_frame, "First frame black"])
                            total_problematic_videos_count += 1
                            break
                        pass  # Allow it to proceed to keypoint extraction, which will likely fail for a black frame

                    #frame_bgr_original = frame.copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

                    # -----------------   DATA AUGMENTATION   -----------------
                    if data_augmentation:
                        # Dir and filenames definitions
                        exercise_output_dir = os.path.join(experiment_base_save_dir, 'first_frames', exercise_name) # e.g. "data/extracted/keypoints/2025-05-29_12-00-00/Rotation"
                        frame = augmentor.augment_data(frame, current_frame, config.currently_plotting, exercise_output_dir, video_name)
                        
                    # -----------------         MODEL   -----------------------
                    frame_process, ratio = model2d.preprocess(frame) # TODO: CHECK IF RGB
                    keypoints_2d, scores_2d = model2d.postprocess(model2d(frame_process), ratio)
                    # -----------------     POSTPROCESSING   ------------------
                    
                    # Temp for holding outputs from filter functions
                    temp_people_dict_output = {}
                    temp_current_means_output = np.array([]) 
                    
                    # 1. In case the model failed and didn't successfully detect people
                    if len(keypoints_2d) == 0:
                        # no processing done on that frame, but the frames are saved for further processing
                        # saved count for the gap
                        videos_missing_keypoints[exercise_name][video_path]["missing_frames"].append(current_frame)

                        if config.currently_plotting:
                            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # frame is RGB, convert to BGR for VideoWriter

                        continue # does not go further, but will interpolate the keypoints for this frame later if the gap is not too big

                    # 2. In case the model successfully detected people - we clean the duplicates and check the gap 
                    clean_keypoints, current_mean_keypoints_from_model = extract_people_per_video(
                                                            keypoints_2d, scores_2d, threshold=same_person_threshold)
                    gap_size_frames = current_frame - last_valid_frame_idx - 1
                    
                    # 2.0.1. In case the keypoints are filtered out - should not happen
                    if clean_keypoints is None:
                        exercise_problematic_videos_count += 1
                        exercise_problematic_videos.append([video_path, current_frame, "Keypoints detected, but filtered out in extract_people_per_video function."])
                        total_problematic_videos.append([video_path, current_frame, "Keypoints detected, but filtered out in extract_people_per_video function."])
                        total_problematic_videos_count += 1
                        break

                    # 2.1. In case there is no prior tracked people, or first frame for tracking
                    if not _people_dict_from_last_filter_success: 
                            temp_people_dict_output, temp_current_means_output = filter_first_frame(
                                current_mean_keypoints_from_model,
                                clean_keypoints,
                                frame_shape,
                                config.filter_to_one_skeleton,
                                normalize_skeleton,
                                random_scaling_factor,
                                current_frame=current_frame
                                )
                    # 2.2. In case the keypoints are not filtered out
                    else:
                        # 2.2.1. In case the gap is 0, we just filter the keypoints
                        if gap_size_frames == 0:
                            temp_people_dict_output, temp_current_means_output = filter_next_frame(
                                _people_dict_from_last_filter_success, # people_dict from last success
                                current_mean_keypoints_from_model,     # candidates from current frame
                                _mean_kps_from_last_filter_success,    # mean_kps state from last success
                                clean_keypoints,
                                config.same_person_threshold,
                                normalize_skeleton,
                                random_scaling_factor,
                                current_frame=current_frame
                            )

                        # 2.2.2. In case the gap is not 0, but not too big - interpolate the keypoints
                        elif 0 < gap_size_frames < config.max_interpolation_gap:
                            temp_people_dict_output, temp_current_means_output = filter_next_frame(
                                    _people_dict_from_last_filter_success, # people_dict from last success
                                    current_mean_keypoints_from_model,     # candidates from current frame
                                    _mean_kps_from_last_filter_success,    # mean_kps state from last success
                                    clean_keypoints,
                                    np.inf, # no checking for distance threshold since we are interpolating and have a big gap # TODO: cleaner
                                    normalize_skeleton,
                                    random_scaling_factor,
                                    current_frame=current_frame
                                )

                            if last_valid_frame_people_dict:
                                interpolated_people_dict = interpolate_keypoints(
                                    last_valid_frame_people_dict, 
                                    temp_people_dict_output,
                                    gap_size_frames,
                                )
                                for interp_dict in interpolated_people_dict:
                                    current_video_frames_data.append(interp_dict)
                
                        # 2.2.3. In case the gap is too big - skip the video
                        else: # gap_size_frames > config.max_interpolation_gap
                            exercise_incomplete_videos.append([video_path, current_frame, total_frames, "Gap size frames exceeds max_interpolation_gap."])
                            total_incomplete_annotated_videos.append([video_path, current_frame, total_frames, "Gap size frames exceeds max_interpolation_gap."])
                            total_incomplete_annotated_videos_count += 1
                            print(f"\nGap size frames ({gap_size_frames}) exceeds max_interpolation_gap "
                                f"({config.max_interpolation_gap})")
                            break
                    
                    # 3. In case the filtering was successful - we have to update the dict and the state
                    if len(temp_people_dict_output) == 0:
                        exercise_problematic_videos_count += 1
                        exercise_problematic_videos.append([video_path, current_frame, "Filtering failed."])
                        total_problematic_videos.append([video_path, current_frame, "Filtering failed."])
                        total_problematic_videos_count += 1
                        break
                        
                    current_video_frames_data.append(temp_people_dict_output)
                    last_valid_frame_people_dict = temp_people_dict_output
                    last_valid_frame_idx = current_frame

                    _people_dict_from_last_filter_success = temp_people_dict_output # to save it for the next iteration, for interpolation
                    _mean_kps_from_last_filter_success = temp_current_means_output
                    
                    # TODO: plot interpolated frames as well! 
                    
                    if config.currently_plotting:
                        # frame is in RGB, plot_keypoints_on_frame_opencv expects RGB and returns RGB
                        plotted_frame_rgb = plot_keypoints_on_frame_opencv(frame.copy(), temp_people_dict_output)
                        out.write(cv2.cvtColor(plotted_frame_rgb, cv2.COLOR_RGB2BGR))
            
            if config.currently_plotting and out.isOpened():
                out.release()
                print(f"\nAnnotated video saved to {annotated_video_path}")
            vid.release()

            if current_video_frames_data:
                # Adding frames for 3d - not from the video but copying
                if config.add_N_frames_before > 0:
                    frame_to_copy = current_video_frames_data[0]
                    current_video_frames_data = [frame_to_copy[0]]*config.add_N_frames_before + current_video_frames_data

                videos_by_exercise[exercise_name][video_path] = current_video_frames_data

                if len(current_video_frames_data) < config.add_N_frames_before and config.add_N_frames_before != 0:
                    warnings.warn(f"{video_path} has less than {config.add_N_frames_before} frames. This should not happen.")
        
        # TODO: CHECK HERE ------------------------------------------
        # After processing all videos for the current EXERCISE, update the pickle file.
        if any(videos_by_exercise[exercise_name].values()):          
            all_exercises_data[exercise_name] = videos_by_exercise[exercise_name]

            with open(pickle_save_path, "wb") as f: 
                pickle.dump(all_exercises_data, f)

            print(f"\nKeypoints for {set_name}, exercise '{exercise_name}' saved/updated in {pickle_save_path}")
        else:
            print(f"\nNo new keypoint data generated for {set_name} set, exercise '{exercise_name}'. Pickle file not updated.")
        # ----------------------------------------------------------------
        if videos_missing_keypoints[exercise_name]:
            with open(missing_keypoints_frames_pickle_path, "wb") as f:
                pickle.dump(videos_missing_keypoints, f)

            with open(missing_keypoints_frames_txt_path, "a") as f:
                f.write(f"\nSet name: {set_name}, Exercise name: {exercise_name}\n")
                for video_path, missing_frames in videos_missing_keypoints[exercise_name].items():
                    f.write(f"{video_path}: {missing_frames}\n")

        # Save exercise-specific logs
        with open(incomplete_videos_txt_path, "a") as f:
            f.write(f"\nSet name: {set_name}, Exercise name: {exercise_name}\n")
            f.write(f"Video path, Frame, Total frames, Error message\n")
            for video_log_entry in exercise_incomplete_videos: 
                f.write(f"{video_log_entry}\n")
        
        with open(problematic_videos_txt_path, "a") as f:
            f.write(f"\nSet name: {set_name}, Exercise name: {exercise_name}\n")
            f.write(f"Video path, Frame, Error message\n")
            for video_log_entry in exercise_problematic_videos: # here we append the file with the total videos. maybe check how to change that
                f.write(f"{video_log_entry}\n")
        
        print(f"\nLogs for exercise '{exercise_name}' ({set_name} set) saved.")
        print(f"Incomplete videos for '{exercise_name}': {len(exercise_incomplete_videos)}")
        print(f"Problematic videos (errors/no keypoints) for '{exercise_name}': {exercise_problematic_videos_count}")
    
    with open(incomplete_videos_pickle_path, "wb") as f:
        pickle.dump(total_incomplete_annotated_videos, f)
    
    with open(problematic_videos_pickle_path, "wb") as f:
            pickle.dump(total_problematic_videos, f)

    print(f"\nLogs for {set_name} set saved.")
    print(f"Total incomplete videos: {total_incomplete_annotated_videos_count}")
    print(f"Total problematic videos (errors/no keypoints): {total_problematic_videos_count}")
    
    if os.path.exists(pickle_save_path):
        return [pickle_save_path]
    else:
        return None

if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        default="src/rg_ai/keypoints_pipeline/config.yaml",
        help="Path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        yaml_config = yaml.safe_load(file).get("extract_keypoints_config")

    config = ExtractKeypointsConfig(**yaml_config)
   
    if not config.keypoints_experiment_name: 
        keypoints_run_identifier = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        keypoints_run_identifier = config.keypoints_experiment_name

    experiment_base_save_dir = os.path.join(os.getenv("KEYPOINT_SAVE_FOLDER"), keypoints_run_identifier)
    os.makedirs(experiment_base_save_dir, exist_ok=True)
    print(f"Experiment base save dir: {experiment_base_save_dir}")

    if not config.train_test_split:
        config.set_names = ["all"] # just for compatibility with the previous code
    start_time = time.time()
    for set_name_iter in config.set_names: 
        for videos_folder_iter in config.videos_folders:
            print(f"Processing {set_name_iter} set from path {videos_folder_iter}")
            output_paths = main(
                experiment_base_save_dir,
                config, 
                set_name_iter,
                videos_folder_iter,
                keypoints_run_identifier,
                master_folder=os.path.basename(videos_folder_iter),
            )
            if output_paths:
                print(f"Successfully processed and saved data for {set_name_iter} from {videos_folder_iter}.")
                # for pth in output_paths:
                #     print(f"  - {pth}")
            else:
                print(f"No data saved for {set_name_iter} from {videos_folder_iter}.")
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds ({time_taken/60} minutes)")
    # ------------------- Save the config ---------------------------
    # NOTE: HERE WE SAVE ONLY THE EXTRACT_KEYPOINTS_CONFIG, NOT THE FULL CONFIG.YAML
    # save time_taken to the config
    yaml_config["time_taken"] = time_taken
    config_copy_path = os.path.join(experiment_base_save_dir, "config.yaml")
    if not os.path.exists(config_copy_path): 
        with open(config_copy_path, 'w') as yamlfile:
            data = yaml.dump(yaml_config, yamlfile)
            print(data)
            print("Write successful")
            yamlfile.close()
    else:
        raise ValueError(f"Config file already exists at {config_copy_path}! Check if you are overwriting the same experiment.")
    
    print(f"Keypoints saved to {experiment_base_save_dir}")
    print("Done.")