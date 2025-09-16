import os

import numpy as np

def get_exercise_name(video_path, master_folder):
    parts = video_path.split(os.sep)
    
    try:
        master_data_index = parts.index(master_folder)  
        exercise_name = parts[master_data_index + 1]  
        return exercise_name
    except Exception as e:
        print(f"Error getting exercise name: {e}")
        return None


def extract_people_per_video(keypoints_2d: np.ndarray, scores_2d: np.ndarray, threshold=50):
    """
    Extract people from the 2D keypoints with multiple people.

    Parameters:
    - keypoints_2d: A list of 2D keypoints for n people.
    - scores_2d: A list of scores for n people.
    - threshold: The threshold for the distance between the mean keypoints of two people. If the distance is less than the threshold, the person with the lower score is removed.

    Returns:
    - frame_keypoints: A list of keypoints for n people.
    - mean_keypoints: The mean keypoints for n people.
    """
    n_people = len(keypoints_2d)
    if n_people == 0:
        return None, None
    
    mean_keypoints = np.mean(keypoints_2d, axis=1)  # get the mean x, y for each person

    # remove duplicates based on threshold and scores
    duplicated_people = set()
    for i in range(n_people):
        for j in range(i + 1, n_people):
            distance = np.linalg.norm(mean_keypoints[i] - mean_keypoints[j])
            if distance < threshold:
                lower_score_idx = i if np.mean(scores_2d[i]) < np.mean(scores_2d[j]) else j
                duplicated_people.add(lower_score_idx)

    # rem duplicates
    filtered_keypoints_2d = [kp for idx, kp in enumerate(keypoints_2d) if idx not in duplicated_people]
    filtered_scores_2d = [sc for idx, sc in enumerate(scores_2d) if idx not in duplicated_people]

    # [x,y,score] for each person
    frame_keypoints = [
        [[x, y, score] for (x, y), score in zip(person_keypoints, person_scores)]
        for person_keypoints, person_scores in zip(filtered_keypoints_2d, filtered_scores_2d)
    ]

    if len(frame_keypoints) == 0:
        return None, None

    mean_keypoints = np.mean(filtered_keypoints_2d, axis=1)

    return frame_keypoints, mean_keypoints

def extract_video_paths(videos_folder, master_folder, filtered_subdirs: list = list()):
    """
    Parameters:
    - VIDEO_PATH: folder

    Returns:
    - video_paths: list of video paths
    """
    # to extract all mp4 from subdirectories using os.walk and keep each folder's name

    video_paths = []
    for root, dirs, files in os.walk(videos_folder):
        for file in files:
            if file.endswith(".mp4"):
                #relative_path = os.path.relpath(root, videos_folder)
                relative_path = get_exercise_name(root, master_folder)

                # if we want to keep only certain subdirectories
                if filtered_subdirs:
                    if relative_path not in filtered_subdirs: # skip the video if not in the filtered subdirs
                        continue

                video_paths.append((os.path.join(root, file), relative_path))
              
    return video_paths

def remove_problematic_videos(current_data, video_paths, problematic_videos):
    """
    Remove problematic videos from the dataset - here they did not have any keypoints in the current_dataset. Remove from both the data and paths.
    """
    count = 0
    for video_path in problematic_videos:
        for exercise_name in current_data:
                if video_path in current_data[exercise_name]:
                    count += 1
                    
                    del current_data[exercise_name][video_path]
                   
                    video_paths.remove((video_path, exercise_name))


    print(f"Removed {count} problematic videos ")

    return current_data, video_paths


def add_frames_for_3d(current_data, n_frames_before=60):
    """
    We need to add n frames before each video for the 3D model (MotionBERT). This is done by copying the first frame 60 times.
    """
    problematic_videos_count = 0
    problematic_videos = []
    for exercise_name in current_data:
        for video_path in current_data[exercise_name]:
            frames = current_data[exercise_name][video_path]
            try:
                current_data[exercise_name][video_path] = [frames[0]]*n_frames_before + frames

            except IndexError:
                print(f"Problematic video: {video_path}")
                problematic_videos.append(video_path)
                problematic_videos_count += 1                   
                continue
                
    print("Problematic videos (for added 3d frames)", problematic_videos_count)

    #with open(os.path.join(current_folder, "problematic_videos", f"problematic_videos-create-datasets_{videos_type}.pkl"), "wb") as f:
    #    pickle.dump(problematic_videos, f)

    return current_data, problematic_videos


def calculate_skeleton_size(keypoints):
    """
    Calculate the skeleton size based on a few keypoints distances (e.g., between shoulders, hips).
    Parameters: 
    - keypoints: List of keypoints for a person

    Returns:
    - The approximate size of the skeleton
    """
    shoulder_distance = np.linalg.norm(np.array(keypoints[5][:2]) - np.array(keypoints[6][:2]))  # between shoulders
    hip_distance = np.linalg.norm(np.array(keypoints[11][:2]) - np.array(keypoints[12][:2]))  # between hips
    nose_ankle_distance = np.linalg.norm(np.array(keypoints[0][:2]) - np.array(keypoints[16][:2]))  # between nose and ankle

    return shoulder_distance + hip_distance + nose_ankle_distance

def scale_keypoints(keypoints, scaling_factor):
    """
    Scale the keypoints by the scaling factor.
    Parameters:
    - keypoints: List of keypoints for a person
    - scaling_factor: Factor by which to scale the keypoints

    Returns:
    - Scaled keypoints
    """
    return [[kp[0] * scaling_factor, kp[1] * scaling_factor, kp[2]] for kp in keypoints]

def normalize_keypoints_for_distance(people_dict, scaling_factor):
    """
    Normalize keypoints for people further away from the camera by scaling them up.
    Parameters:
    - people_dict: Dictionary containing keypoints and mean keypoints for each person.
    - reference_size: The reference skeleton size to compare against.

    Returns:
    - Updated people_dict
    """
    for person_id, person_data in people_dict.items():
        keypoints = person_data["keypoints"]
        #current_skeleton_size = calculate_skeleton_size(keypoints)
        
        # scaling factor based on reference skeleton size
        #scaling_factor = reference_size / current_skeleton_size if current_skeleton_size > 0 else 1.0
        # while still being in the frame     
        scaled_keypoints = scale_keypoints(keypoints, scaling_factor)

        # if any keypoint is outside the frame, do we need to adjust the keypoints
        person_data["keypoints"] = scaled_keypoints

    return people_dict


def filter_first_frame(current_mean_keypoints: np.ndarray,
                        clean_keypoints: np.ndarray,
                        frame_shape: tuple, 
                        filter_to_one_skeleton: str="biggest", 
                        normalize_skeleton: bool=False,
                        random_scaling_factor: float=1.0,
                        current_frame: int=None,
                        ):
    """
    Filter the first frame to keep only one person and normalize the skeleton size.
    """
    next_person_id = 0
    people_dict = {}
    num_people = len(current_mean_keypoints)
    ############### 1. KEEP ONLY ONE PERSON ###############
    if num_people > 1:
        # keep the person with the biggest skeleton size - so between ear and foot
        keep_skeleton_idx = 0
        if filter_to_one_skeleton=="biggest":
            biggest_skeleton_size = 0
            for i in range(num_people):
                nose = clean_keypoints[i][0]
                foot = clean_keypoints[i][16]
                skeleton_size = np.linalg.norm(np.array(nose[:2]) - np.array(foot[:2]))
                if skeleton_size > biggest_skeleton_size:
                    biggest_skeleton_size = skeleton_size
                    keep_skeleton_idx = i

        elif filter_to_one_skeleton=="most_central":
            most_central_skeleton_distance = np.inf
            
            frame_height, frame_width = frame_shape
            central_video_coordinates = np.array([frame_width/2, frame_height/2])

            for i in range(num_people):
                skeleton_distance = np.linalg.norm(np.array(current_mean_keypoints[i]) - central_video_coordinates)
                if skeleton_distance < most_central_skeleton_distance:
                    most_central_skeleton_distance = skeleton_distance
                    keep_skeleton_idx = i

        clean_keypoints = [clean_keypoints[keep_skeleton_idx]]
        current_mean_keypoints = [current_mean_keypoints[keep_skeleton_idx]]

    #################################################################
    
    # init people dict
    for kp in current_mean_keypoints:
        people_dict[next_person_id] = {
            "mean_keypoints": kp,
            "keypoints": clean_keypoints[next_person_id],
            "current_frame": current_frame,
        }
        next_person_id += 1  
    
    ########### 2. SCALING AND NORMALIZATION ###########
    #  especially needed for smaller skeleton sizes - RANDOM RESCALE OR FIXED ?
    if normalize_skeleton: 
        people_dict = normalize_keypoints_for_distance(people_dict, random_scaling_factor)


    return people_dict, current_mean_keypoints

def filter_next_frame(
        people_dict: dict, 
        current_mean_keypoints: np.ndarray,
        previous_mean_keypoints: np.ndarray,
        clean_keypoints: np.ndarray,
        same_person_threshold: int,
        normalize_skeleton: bool=False,
        random_scaling_factor: float=1.0,
        current_frame: int=None,
        ):
    # checking with the previous mean keypoints and bind the new keypoints to the closest old ones
    num_people = len(current_mean_keypoints)

    unmatched_people = set(people_dict.keys())  # Track unmatched people
    new_people_dict = {}

    ########### HANDLING MULTIPLE PEOPLE IN INDIVIDUAL ATHLETE VIDEOS ###############
    if num_people > 1:
        distances = {}
        # keep the person with the closest mean to the previous mean keypoints
        for i, mean_kp in enumerate(current_mean_keypoints):
            distances[i] = [np.linalg.norm(mean_kp - previous_mean_keypoints[pid])
                for pid in unmatched_people]

        closest_person_id = min(distances, key=distances.get)

        for pid in unmatched_people:
            new_people_dict[pid] = {
                "mean_keypoints": current_mean_keypoints[closest_person_id],
                "keypoints": clean_keypoints[closest_person_id],
                "current_frame": current_frame,
            }

    ###################################################################################
    # if there are less or more current_mean_keypoints than previous_mean_keypoints, we need to handle this
    ####### HANDLING MULTIPLE PEOPLE - SWITCHING KEYPOINTS AND PEOPLE GOING OUT OF FRAME - IN NOISE VIDEOS #########
    else: 
        for i, mean_kp in enumerate(current_mean_keypoints):
            distances = {
                pid: np.linalg.norm(mean_kp - people_dict[pid]["mean_keypoints"])
                for pid in unmatched_people
            }
            closest_person_id = min(distances, key=distances.get)

            if distances[closest_person_id] < same_person_threshold:
                new_people_dict[closest_person_id] = {
                    "mean_keypoints": mean_kp,
                    "keypoints": clean_keypoints[i],
                    "current_frame": current_frame,
                }
                # unmatched_people.remove(closest_person_id)

            else:
                # TODO: can implement that we add a person if a new one appears
                pass

    #####################################################################################
    # TODO: remove the person from the dict if not moving for the last n frames
    people_dict = new_people_dict

    # NORMALIZE PEOPLE SIZE - RANDOM RESCALE OR FIXED ?
    if normalize_skeleton:
        people_dict = normalize_keypoints_for_distance(people_dict, random_scaling_factor)
    
    return people_dict, current_mean_keypoints

def interpolate_keypoints(prev_keypoints, next_keypoints, num_frames):
    """
    Interpolate keypoints between two frames.
    Args:
        prev_keypoints (dict): Dictionary of keypoints from the previous valid frame
        next_keypoints (dict): Dictionary of keypoints from the next valid frame
        num_frames (int): Number of frames to interpolate
    Returns:
        list: List of interpolated keypoint dictionaries
    """
    interpolated_frames = []
    
    for frame_idx in range(1, num_frames+1): # TODO; check if num_frames + 1 is correct
        alpha = frame_idx / (num_frames+1) 
        interpolated_dict = {}
        
        for person_id in prev_keypoints.keys():
            if person_id in next_keypoints:
                prev_kps = prev_keypoints[person_id]["keypoints"]
                next_kps = next_keypoints[person_id]["keypoints"]

                prev_frame = prev_keypoints[person_id]["current_frame"]
                next_frame = next_keypoints[person_id]["current_frame"]

                if next_frame - prev_frame != num_frames+1:
                    raise ValueError(f"The number of frames between the previous and next valid frame is not equal to the number of frames to interpolate. "
                                     f"Previous frame: {prev_frame}, Next frame: {next_frame}, Number of frames to interpolate: {num_frames}")
                
                interpolated_kps = []
                
                # Linear interpolation
                for i, prev_kp in enumerate(prev_kps):
                    next_kp = next_kps[i]
                    interpolated_x = prev_kp[0] + alpha * (next_kp[0] - prev_kp[0])
                    interpolated_y = prev_kp[1] + alpha * (next_kp[1] - prev_kp[1])
                    interpolated_kps.append([interpolated_x, interpolated_y, prev_kp[2]]) # previous score
                
                mean_kp = np.mean(interpolated_kps, axis=0)
                
                interpolated_dict[person_id] = {
                    "mean_keypoints": np.array(mean_kp[:2]),
                    "keypoints": interpolated_kps,
                    "current_frame": prev_frame + frame_idx,
                }
        
        interpolated_frames.append(interpolated_dict)
    
    return interpolated_frames