import json
import os
import subprocess
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings

@dataclass
class VideoProperties:
    duration: float
    fps: Optional[float] = None


@dataclass
class Annotation:
    start_time: float
    end_time: float
    movement_type: str
    subtype: str

@dataclass
class VideoProcessorConfig:
    annotations_folder: str
    videos_folder: str
    output_folder: str
    padding_frames_start: int = 0
    padding_frames_end: int = 0
    extract_inbetween_segments: bool = False
    extract_inbetween_segments_n_frames_threshold: int = 0
    verbose: bool = False


@dataclass
class InbetweenSegment:
    """Represents a segment between two close annotations."""
    current_anno: Annotation
    next_anno: Annotation
    start_time: float
    end_time: float
    type1: str
    type2: str
    subtype1: str
    subtype2: str
    
    @property
    def segment_name(self) -> str:
        """Generate segment name for inbetween segments."""
        if self.type1 == self.type2:
            return f"{self.type1}-{self.type1}_inbetween"
        return f"{self.type1}-{self.type2}_inbetween"


class VideoProcessor:
    """Handles video annotation processing and segment cutting."""
    
    def __init__(self, 
                 config: VideoProcessorConfig,
                 ):
        
        self.annotations_folder = config.annotations_folder
        self.videos_folder = config.videos_folder
        self.output_folder = config.output_folder
        self.padding_frames_start = config.padding_frames_start
        self.padding_frames_end = config.padding_frames_end
        self.extract_inbetween_segments_n_frames_threshold = config.extract_inbetween_segments_n_frames_threshold
        self.extract_inbetween_segments = config.extract_inbetween_segments
        self.verbose = config.verbose
    
    def get_video_properties(self, video_path: str) -> Optional[VideoProperties]:
        """Get video duration and FPS using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,avg_frame_rate,codec_type',
            '-show_entries', 'format=duration', '-of', 'json', video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                raise RuntimeError("ffprobe not found. Install ffmpeg and add to PATH.")
            warnings.warn(f"Failed to get video properties for {video_path}")
            return None
        
        duration = None
        if 'format' in data and 'duration' in data['format']:
            try:
                duration = float(data['format']['duration'])
            except (ValueError, TypeError):
                warnings.warn(f"Invalid duration in {video_path}")
                return None
        
        if duration is None:
            warnings.warn(f"No duration found for {video_path}")
            return None
        
        fps = None
        if 'streams' in data and data['streams']:
            for stream in data['streams']:
                if stream.get('codec_type') == 'video':
                    fps_str = stream.get('r_frame_rate') or stream.get('avg_frame_rate')
                    if fps_str and fps_str != "0/0":
                        try:
                            if '/' in fps_str:
                                num, den = fps_str.split('/')
                                fps = float(num) / float(den) if float(den) > 0 else None
                            else:
                                fps = float(fps_str)
                            if fps and fps <= 0:
                                fps = None
                        except ValueError:
                            pass
                    break
        
        return VideoProperties(duration=duration, fps=fps)
    
    def cut_video_segment(self, video_path: str, start: float, end: float, 
                         output_path: str) -> bool:
        """Cut a video segment using ffmpeg."""
        if end <= start:
            return False
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            'ffmpeg', '-i', video_path, '-ss', str(start), '-to', str(end),
            '-c', 'copy', '-y', '-loglevel', 'error', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                raise RuntimeError("ffmpeg not found. Install ffmpeg and add to PATH.")
            warnings.warn(f"Failed to cut segment {start}-{end} from {video_path}")
            return False
    
    def load_annotations(self) -> Dict[str, List[Annotation]]:
        """Load and parse all annotation files."""
        video_annotations = {}
        
        # JSON files in all subfolders
        annotation_files = []
        for root, dirs, files in os.walk(self.annotations_folder):
            for file in files:
                if file.endswith('.json'):
                    annotation_files.append(os.path.join(root, file))
        
        if not annotation_files:
            warnings.warn(f"No annotation files found in {self.annotations_folder} or its subfolders")
            return {}
        
        for filepath in annotation_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_annotations = self._parse_annotation_data(data)
                
                for video_name, annotations in file_annotations.items():
                    if video_name not in video_annotations:
                        video_annotations[video_name] = []
                    video_annotations[video_name].extend(annotations)
                    
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
        
        return video_annotations
    
    def _parse_annotation_data(self, data) -> Dict[str, List[Annotation]]:
        """Parse annotation data from JSON."""
        annotations = {}
        
        if isinstance(data, list):
            # all annotations for one video
            video_name = None
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                current_video = item.get("video")
                if not current_video:
                    continue
                
                if video_name is None:
                    video_name = current_video
                elif video_name != current_video:
                    warnings.warn("Multiple videos in single annotation file")
                    break
                
                annotation = self._create_annotation(item)
                if annotation:
                    if video_name not in annotations:
                        annotations[video_name] = []
                    annotations[video_name].append(annotation)
        
        elif isinstance(data, dict):
            # video_name -> annotations
            for video_name, video_data in data.items():
                if not isinstance(video_data, dict):
                    continue
                
                video_annotations = []
                for anno_data in video_data.values():
                    annotation = self._create_annotation(anno_data)
                    if annotation:
                        video_annotations.append(annotation)
                
                if video_annotations:
                    annotations[video_name] = video_annotations
        
        return annotations
    
    def _create_annotation(self, data: dict) -> Optional[Annotation]:
        """Create annotation from dict data."""
        try:
            return Annotation(
                start_time=float(data["start_time"]),
                end_time=float(data["end_time"]),
                movement_type=str(data["movement_type"]).replace("/", "_").replace(" ", "_"),
                subtype=str(data["subtype"]).replace("/", "_").replace(" ", "_")
            )
        except (KeyError, ValueError, TypeError):
            return None
    
    def _apply_padding(self, annotation: Annotation, properties: VideoProperties, inbetween_segments: List[InbetweenSegment]) -> Tuple[float, float]:
        """Apply frame padding to annotation times."""

        start_time = annotation.start_time
        end_time = annotation.end_time

        if properties.fps and properties.fps > 0 and (self.padding_frames_start > 0 or self.padding_frames_end > 0):
            start_padding = self.padding_frames_start / properties.fps
            end_padding = self.padding_frames_end / properties.fps
            
            start_time = max(0.0, annotation.start_time - start_padding) # 10 -> 5
            end_time = min(properties.duration, annotation.end_time + end_padding) # 15-> 20
        
        # No padding in the inbetween segment parts
        for seg in inbetween_segments:
            if annotation == seg.current_anno:
                end_time = annotation.end_time
            
            if annotation == seg.next_anno:
                start_time = seg.current_anno.end_time # this means that we immediately switch to the next annotation
        
        return start_time, end_time
    
    def _detect_inbetween_segments(self, annotations: List[Annotation], 
                                 properties: VideoProperties) -> List[InbetweenSegment]:
        """Detect and create inbetween segments for close annotations."""
        if self.extract_inbetween_segments_n_frames_threshold <= 0 or not properties.fps or properties.fps <= 0:
            return []
        
        inbetween_segments = []
        time_threshold = self.extract_inbetween_segments_n_frames_threshold / properties.fps
        
        for i in range(len(annotations) - 1):
            current_anno = annotations[i]
            next_anno = annotations[i + 1]
            
            gap_duration = next_anno.start_time - current_anno.end_time
            
            if 0 < gap_duration <= time_threshold:
                # In case of different movement types, no need to create inbetween segment
                if current_anno.movement_type != next_anno.movement_type:
                     continue
                
                inbetween_segment = InbetweenSegment(
                    current_anno=current_anno,
                    next_anno=next_anno,
                    start_time=current_anno.end_time,
                    end_time=next_anno.start_time,
                    type1=current_anno.movement_type,
                    type2=next_anno.movement_type,
                    subtype1=current_anno.subtype,
                    subtype2=next_anno.subtype
                )
                inbetween_segments.append(inbetween_segment)
        
        return inbetween_segments
    
    def _process_inbetween_segments(self, video_name: str, video_path: str, 
                                  inbetween_segments: List[InbetweenSegment]) -> List[Tuple[float, float]]:
        """Process and cut inbetween segments."""
        used_segments = []
        
        for idx, segment in enumerate(inbetween_segments):
            target_dir = os.path.join(self.output_folder, "Other", segment.segment_name)
            os.makedirs(target_dir, exist_ok=True)
            clip_name = f"{os.path.splitext(video_name)[0]}_inbetween{idx}_{segment.segment_name}.mp4"
            output_path = os.path.join(target_dir, clip_name)
            
            if self.cut_video_segment(video_path, segment.start_time, segment.end_time, output_path):
                used_segments.append((segment.start_time, segment.end_time))
                if self.verbose:
                    print(f"  Created inbetween segment: {segment.segment_name} "
                          f"({segment.start_time:.2f}s - {segment.end_time:.2f}s)")
        
        return used_segments
    
    def _create_other_segments(self, video_name: str, video_path: str, 
                              used_segments: List[Tuple[float, float]], 
                              duration: float):
        """Create 'Other' segments for unannotated video parts."""
        used_segments.sort()
        other_dir = os.path.join(self.output_folder, "Other")
        
        current_time = 0.0
        other_idx = 0
        
        for start, end in used_segments:
            if start > current_time and start - current_time > 0.1:
                # before current segment
                clip_name = f"{os.path.splitext(video_name)[0]}_seg{other_idx}_Other_GENERAL.mp4"
                output_path = os.path.join(other_dir, clip_name)
                
                if self.cut_video_segment(video_path, current_time, start, output_path):
                    other_idx += 1
            
            current_time = max(current_time, end)
        
        # after last segment
        if current_time < duration and duration - current_time > 0.1:
            clip_name = f"{os.path.splitext(video_name)[0]}_seg{other_idx}_Other_GENERAL.mp4"
            output_path = os.path.join(other_dir, clip_name)
            self.cut_video_segment(video_path, current_time, duration, output_path)
    
    def process(self):
        """Main processing function."""
        print(f"Processing videos from {self.videos_folder}")
        print(f"Output folder: {self.output_folder}")
        if self.extract_inbetween_segments and self.extract_inbetween_segments_n_frames_threshold > 0:
            print(f"Inbetween segments enabled: {self.extract_inbetween_segments_n_frames_threshold} frames threshold")
        
        video_annotations = self.load_annotations()
        if not video_annotations:
            warnings.warn("No valid annotations found")
            return
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        for video_name, annotations in video_annotations.items():
            video_path = os.path.join(self.videos_folder, video_name)
            
            if not os.path.exists(video_path):
                warnings.warn(f"Video not found: {video_path}")
                continue
  
            properties = self.get_video_properties(video_path)
            if not properties:
                warnings.warn(f"Cannot process {video_name} - no video properties")
                continue
            
            print(f"Processing {video_name} ({properties.duration:.1f}s, FPS: {properties.fps or 'unknown'})")
            
            annotations.sort(key=lambda x: x.start_time)

            inbetween_segments = []
            if self.extract_inbetween_segments:
                inbetween_segments = self._detect_inbetween_segments(annotations, properties)
            
            used_segments = []

            for idx, annotation in enumerate(annotations):
                start_time, end_time = self._apply_padding(annotation, properties, inbetween_segments)
                
                if end_time <= start_time or start_time >= properties.duration:
                    continue
                
                end_time = min(end_time, properties.duration)
                
                target_dir = os.path.join(self.output_folder, annotation.movement_type, annotation.subtype)
                clip_name = f"{os.path.splitext(video_name)[0]}_seg{idx}_{annotation.movement_type}_{annotation.subtype}.mp4"
                output_path = os.path.join(target_dir, clip_name)
                
                if self.cut_video_segment(video_path, start_time, end_time, output_path):
                    used_segments.append((start_time, end_time))
            
            #inbetween_used_segments = self._process_inbetween_segments(video_name, video_path, inbetween_segments)
            #used_segments.extend(inbetween_used_segments)
            
            self._create_other_segments(video_name, video_path, used_segments, properties.duration)
        
        print("Processing complete")


if __name__ == "__main__":
    # Config ---------------------
    config = VideoProcessorConfig(
        annotations_folder = "data/raw/annotations", #/Annotatios Tokyo Q pack 1",
        videos_folder = "data/raw/videos/OG_2020_Tokyo_Qualifications1",
        output_folder = "data/processed_videos/Annotatios Tokyo Q pack 1 2 and four of 5 padding 121 start 121 end",
        padding_frames_start = 121,
        padding_frames_end = 121,
        extract_inbetween_segments = False, 
        extract_inbetween_segments_n_frames_threshold = 0,
        verbose = True
    )
    # -----------------------------
    if os.path.exists(config.output_folder):
        warnings.warn(f"Output folder '{config.output_folder}' already exists")
        overwrite = input(f"Do you want to overwrite the output folder '{config.output_folder}'? (y/n): ")
        if overwrite.lower() not in ['y', 'yes', '']:
            exit(1)
        else:
            pass

    if not os.path.isdir(config.videos_folder):
        warnings.warn(f"Videos folder '{config.videos_folder}' does not exist")
        exit(1)
    elif not os.path.isdir(config.annotations_folder):
        warnings.warn(f"Annotations folder '{config.annotations_folder}' does not exist")
        exit(1)
    
    processor = VideoProcessor(config = config)
    processor.process()


