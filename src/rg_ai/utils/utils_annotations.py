import os
import json
import glob
import warnings
from typing import List, Dict


def get_ground_truth_from_annotations(
    current_time: float, 
    annotations: List[Dict], 
    use_sublabels: bool = False,
    use_consolidation: bool = False,
    consolidation_map: dict = None,
    ) -> str:
    """Get ground truth from annotations"""
    for ann in annotations:
        start_time = ann.get('start_time', 0)
        end_time = ann.get('end_time', float('inf'))
        
        if start_time <= current_time <= end_time:
            main_label = ann['movement_type']
            if use_sublabels:
                sublabel = ann['subtype']

                if use_consolidation and consolidation_map:
                    sublabel = sublabel.replace(" ", "_")
                    sublabel = consolidation_map.get(main_label, {}).get(sublabel)

                if sublabel == "GENERAL":
                    return f"{main_label}"
                # add _ instead of spaces
                if sublabel is not None:
                    sublabel = sublabel.replace(" ", "_")
                    return f"{main_label}_{sublabel}"
                else:
                    return main_label
            else:
                return main_label
            
    return "Other"

def get_label_from_filename(
    filename: str, 
    main_label_index: int = 0,                     
    use_sublabels: bool = False, 
    joined_label_dict: dict = None,
    use_consolidation: bool = False, 
    consolidation_map: dict = None,
    other_check: bool = False, 
    fallback_general: bool = False,
    get_only_sublabel: bool = False,
    check_person_suffix: bool = True,
    ) -> str:
    """
    Unified function to extract labels from filenames.
    
    Args:
        filename: The filename to extract label from
        main_label_index: Index position to extract main label from split filename (0 for embedding files, 2 for video names)
        use_sublabels: Whether to use sublabel extraction
        joined_label_dict: Dictionary mapping main labels to their sublabels
        use_consolidation: Whether to apply label consolidation
        consolidation_map: Mapping for consolidated labels
        other_check: Whether to check for "Other" category first
        fallback_general: Whether to fallback to "GENERAL" instead of raising error
    
    Returns:
        Extracted label string
    """
    # checks for "Other" category first if enabled
    if other_check and ("_Other_" in filename or "_other" in filename):
        return "Other"
    
    main_label = filename.split("_")[main_label_index]
    
    if not use_sublabels or not joined_label_dict:
        return main_label
    
    sublabels = joined_label_dict.get(main_label, [])
    if not sublabels:
        return main_label
    
    # sublabels by length (longest first) for better matching
    sorted_sublabels = sorted(sublabels, key=len, reverse=True)
    
    for sublabel in sorted_sublabels:
        check_subtype_suffix = f"{main_label}_{sublabel}_person" if check_person_suffix else f"{main_label}_{sublabel}"
        if fallback_general and sublabel == "GENERAL":
            continue
        
        if check_subtype_suffix in filename:
            if use_consolidation and consolidation_map:
                consolidated = consolidation_map.get(main_label, {}).get(sublabel)
                if consolidated == "GENERAL":
                    return f"{main_label}"
                return f"{main_label}_{consolidated}" if consolidated else f"{main_label}_{sublabel}"

            if sublabel == "GENERAL":
                return f"{main_label}"
            return f"{main_label}_{sublabel}"
    
    if fallback_general:
        return "GENERAL"
    
    raise ValueError(f"Sublabel for {filename} not found in joined_label_dict")

def crawl_annotations_for_video(video_name: str, annotations_folder: str) -> List[Dict]:
    """
    Crawl all JSON files in subfolders and extract annotations for a specific video.
    Expected format: {video_name: {annotation_1: {...}, annotation_2: {...}, ...}}
    
    Args:
        video_name (str): Name of the video file (with or without extension)
        annotations_folder (str): Path to folder containing annotation JSON files
        
    Returns:
        List[Dict]: List of annotations for the video
        
    Raises:
        ValueError: If more than one annotation set is found for the same video
    """
    if not os.path.exists(annotations_folder):
        warnings.warn(f"Annotations folder not found: {annotations_folder}")
        return []
    
    # video name for matching
    video_base_name = os.path.splitext(video_name)[0]
    video_name_with_ext = video_name.name
    
    # finding annotation JSON files recursively (pattern: annotations_*.json)
    json_files = glob.glob(os.path.join(annotations_folder, "**/**.json"), recursive=True)
    all_annotations = []
    sources_found = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # {video_name: {annotation_1: {...}, annotation_2: {...}, ...}, ...}
            for video_key, video_annotations in data.items():
                # TODO: check names here
                if (video_key == video_name_with_ext or 
                    os.path.splitext(video_key)[0] == video_base_name):
                    
                    # list format of numbered annotations (so annnotation_1 is the first annotation in the list and so on...)
                    annotations = []
                    for ann_key, ann_data in video_annotations.items():
                        if ann_key.startswith('annotation_'):
                            ann_data_copy = ann_data.copy()
                            ann_data_copy['annotation_id'] = ann_key
                            ann_data_copy['video'] = video_key
                            ann_data_copy['source_file'] = json_file
                            annotations.append(ann_data_copy)
                    
                    if annotations:
                        all_annotations.extend(annotations)
                        sources_found.append(json_file)
                        #break 
                    
        except (json.JSONDecodeError, IOError) as e:
            warnings.warn(f"Error reading JSON file {json_file}: {e}")
            continue
    
    if len(sources_found) > 1:
        raise ValueError(
            f"Multiple annotation sources found for video '{video_name}': {sources_found}. "
            f"Please ensure only one annotation file contains data for this video."
        )
    
    if not all_annotations:
        warnings.warn(f"No annotations found for video '{video_name}' in {annotations_folder}")
    
    return all_annotations