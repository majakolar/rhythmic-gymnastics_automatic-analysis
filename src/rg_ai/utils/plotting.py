import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict
import pickle
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from rg_ai.utils.skeleton import coco17, h36m


def plot_keypoints_on_frame(frame, current_data, current_frame, exercise_name, video_profile, video_path, video_name):
    """
    Plot keypoints on a frame.

    Parameters:
    - frame: The video frame (numpy array) to plot keypoints on.
    - current_data: A dictionary containing keypoint coordinates, labels, and scores.
    - current_frame: The current frame number.
    - exercise_name: The name of the exercise.
    - video_profile: The profile of the video.
    - video_path: The path of the video.
    """
    base_save_dir = os.getenv("KEYPOINT_SAVE_FOLDER")
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # frame to RGB

    frame_keypoints = current_data[exercise_name][video_profile][video_path][-1]
    # frame_idx = len(current_data[exercise_name][video_profile][video_path]) # switched with current_frame

    for person_idx in frame_keypoints.keys():
        for kp in frame_keypoints[person_idx]["keypoints"]:
            x = kp[0]
            y = kp[1]
            score = kp[2]

            colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "cyan", "magenta"]
            person_color = colors[person_idx % len(colors)]
            plt.scatter(x, y, color=person_color, s=40, label=f"Score: {score:.2f}")
            plt.text(x + 5, y, f"{score:.2f}", color="white", fontsize=5)

    plt.axis("off")
    plt.savefig(
        os.path.join(
            base_save_dir,
            os.path.join(
                "fig",
                f"keypoints_on_frame_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{video_name}_{current_frame}.png",
            ),
        )
    )
    plt.show()


def plot_keypoints_on_frame_opencv(frame, people_dict, output_path=None):
    """
    Plot keypoints on a frame using OpenCV.
    
    Args:
        frame: numpy array, the video frame to plot on
        people_dict: dict, dictionary containing keypoints for each person
        output_path: str, optional path to save the frame. If None, frame is not saved
    
    Returns:
        frame: numpy array, the frame with keypoints drawn on it
    """
    head_color = (255, 0, 0)  
    hands_color = (0, 255, 0) 
    legs_color = (0, 0, 255)

    # SANITY CHECK IF THERE IS ONLY ONE PERSON IN THE FRAME - should be removed in preprocessing
    if len(people_dict) > 1:
        raise KeyError(f"More than one person detected in frame")

    for person_id, person_keypoints in people_dict.items():
        for i, keypoint in enumerate(person_keypoints["keypoints"]):
            x, y = int(keypoint[0]), int(keypoint[1])
            if i <= 4:  # head 
                color = head_color
            elif 5 <= i <= 10:  # hands 
                color = hands_color
            elif 11 <= i <= 16:  # legs 
                color = legs_color
            else:
                color = (255, 255, 255)

            cv2.circle(frame, (x, y), 3, color, -1)
    
    if output_path:
        cv2.imwrite(output_path, frame)
    
    return frame



def draw_mmpose(img, keypoints, scores, keypoint_info, skeleton_info, kpt_thr=0.5, radius=2, line_width=2):
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info["color"])
        link_dict[kpt_info["name"]] = kpt_info["id"]

        kpt = keypoints[i]

        if radius > 0:
            if vis_kpt[i]:
                img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), kpt_color, -1)
                
    if line_width > 0:
        for i, ske_info in skeleton_info.items():
            link = ske_info["link"]
            pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

            if vis_kpt[pt0] and vis_kpt[pt1]:
                link_color = ske_info["color"]
                kpt0 = keypoints[pt0]
                kpt1 = keypoints[pt1]

                img = cv2.line(
                    img, (int(kpt0[0]), int(kpt0[1])), (int(kpt1[0]), int(kpt1[1])), link_color, thickness=line_width
                )

    return img


def draw_skeleton(img, keypoints, scores, openpose_skeleton=False, kpt_thr=0.5, radius=2, line_width=2):
    num_keypoints = keypoints.shape[1]
    #print(f"num_keypoints: {num_keypoints}")
    #print(f"keypoins shape: {keypoints.shape}")

    if openpose_skeleton:
        print(f"num_keypoints: {num_keypoints}")
        print(f"keypoins shape: {keypoints.shape}")

        raise NotImplementedError
    else:
        if num_keypoints == 17:
            skeleton = "coco17"
        else:
            print(f"num_keypoints: {num_keypoints}")
            print(f"keypoins shape: {keypoints.shape}")

            raise NotImplementedError

    keypoint_info = coco17["keypoint_info"]
    skeleton_info = coco17["skeleton_info"]

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    if skeleton in ["coco17"]:
        for i in range(num_instance):
            img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info, skeleton_info, kpt_thr, radius, line_width)
    else:
        raise NotImplementedError
    return np.array(img)

def draw_2d_skeleton_blazepose(frame, keypoints_2d):
    """Draw 2D skeleton on the frame using BlazePose connections"""
    POSE_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Torso
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
        (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        # Right arm  
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        # Left leg
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
    ]
    
    if keypoints_2d is None or len(keypoints_2d) == 0:
        return frame
        
    frame_copy = frame.copy()
    
    for i, point in enumerate(keypoints_2d):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
           
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
            start_point = keypoints_2d[start_idx][:2].astype(int)
            end_point = keypoints_2d[end_idx][:2].astype(int)
            cv2.line(frame_copy, tuple(start_point), tuple(end_point), (0, 255, 0), 2)
                
    return np.array(frame_copy)

def draw3d(img, keypoints_3d):
    scale_factor = 600
    theta_y = np.radians(0)  
    theta_x = np.radians(0)
    theta_z = np.radians(20)
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])

    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for keypoint in keypoints_3d:

        mean_x = np.mean([kp[0] for kp in keypoint])
        mean_z = np.mean([kp[2] for kp in keypoint])

        max_z = max(point[2] for point in keypoint)
        min_z = min(point[2] for point in keypoint)

        scale_factor = 300

        rotated_keypoints = keypoint @ (Rz @ (Rx @ Ry.T))
        x, y, z = rotated_keypoints.T

        colors = ((z - min_z) / (max_z - min_z) * 255).astype(np.uint8)

        x2d = ((x - mean_x) * scale_factor + img.shape[1] // 2).astype(int)
        z2d = ((-z + mean_z) * scale_factor + img.shape[0] // 2).astype(int)

        for xi, zi, color in zip(x2d, z2d, colors):
            color = int(color)
            cv2.circle(image, (xi, zi), 5, (color, 0, 255 - color), -1)

    return image


def draw_id(
    img: np.ndarray, ids: list[str], keypoints: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.5, y_offset: int = 60
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    for i in range(keypoints.shape[0]):
        person_keypoints = keypoints[i]
        vis_kpt = [s >= kpt_thr for s in scores[i]]
        if (len(vis_kpt) == 0) or (not np.any(vis_kpt)):
            continue
        # center x, min y - smt
        min_x, max_x = int(person_keypoints[vis_kpt, 0].min()), int(person_keypoints[vis_kpt, 0].max())
        min_y = int(person_keypoints[vis_kpt, 1].min())

        center_x = (min_x + max_x) // 2

        (text_width, text_height), _ = cv2.getTextSize(ids[i], font, font_scale, thickness)
        text_x = center_x - (text_width // 2)
        text_y = max(min_y - y_offset - (text_height // 2), 0)

        cv2.putText(img, ids[i], (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

    return img

def setup_3d_plot_improved(ax, keypoints_3d=None, padding_factor=0.2, fixed_limits=None):
    """
    Set up a 3D matplotlib plot with proper axes and viewing angle, optimized for skeleton size.
    
    Args:
        ax: matplotlib 3D axes object
        keypoints_3d: np.ndarray of shape (17, 3) with 3D keypoint coordinates
        padding_factor: float, factor to add padding around the skeleton (0.2 = 20% padding)
    """
    ax.clear()

    if fixed_limits is not None:
        ax.set_xlim3d(fixed_limits[0])
        ax.set_ylim3d(fixed_limits[1])
        ax.set_zlim3d(fixed_limits[2])

    elif keypoints_3d is not None and len(keypoints_3d) > 0:
        valid_mask = ~(np.isnan(keypoints_3d).any(axis=1) | np.isinf(keypoints_3d).any(axis=1))
        if np.any(valid_mask):
            valid_keypoints = keypoints_3d[valid_mask]
            
            x_min, x_max = valid_keypoints[:, 0].min(), valid_keypoints[:, 0].max()
            y_min, y_max = valid_keypoints[:, 1].min(), valid_keypoints[:, 1].max()
            z_min, z_max = valid_keypoints[:, 2].min(), valid_keypoints[:, 2].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            x_padding = x_range * padding_factor
            y_padding = y_range * padding_factor
            z_padding = z_range * padding_factor
            
            ax.set_xlim3d([x_min - x_padding, x_max + x_padding])
            ax.set_ylim3d([y_min - y_padding, y_max + y_padding])
            ax.set_zlim3d([z_min - z_padding, z_max + z_padding])
        else:
            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-1, 1])
    else:
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    
    ax.view_init(elev=15, azim=45)
    
    ax.set_box_aspect([1, 1, 1])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray') 
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)


def get_h36m_connections_from_skeleton():
    """Get H36M skeleton connections from the skeleton definition."""
    connections = []
    keypoint_info = h36m["keypoint_info"]
    skeleton_info = h36m["skeleton_info"]
    
    name_to_id = {info["name"]: info["id"] for info in keypoint_info.values()}
    
    # skeleton links to (start_idx, end_idx)
    for ske_info in skeleton_info.values():
        link = ske_info["link"]
        start_name, end_name = link[0], link[1]
        start_idx = name_to_id[start_name]
        end_idx = name_to_id[end_name]
        connections.append((start_idx, end_idx))
    
    return connections


def draw_3d_skeleton_improved(ax, keypoints_3d, scores=None, confidence_threshold=0.3, 
                             point_size=100, line_width=3, fixed_limits=None):
    """
    Draw a clean 3D skeleton using logical connections and colors from H36M definition. MotionBERT uses H36M.
    
    Args:
        ax: matplotlib 3D axes object
        keypoints_3d: np.ndarray of shape (17, 3) with 3D keypoint coordinates  
        scores: np.ndarray of shape (17,) with confidence scores (optional)
        confidence_threshold: float, minimum confidence to show a joint
        point_size: int, size of the points
        line_width: int, width of the skeleton lines
        fixed_limits: np.array of shape (3, 2) with min and max values for each axis
    """
    if keypoints_3d is None or len(keypoints_3d) == 0:
        return
    
    setup_3d_plot_improved(ax, keypoints_3d, padding_factor=0.15, fixed_limits=fixed_limits)
    
    keypoint_info = h36m["keypoint_info"]
    skeleton_info = h36m["skeleton_info"]

    #joint_pairs = get_h36m_connections_from_skeleton()
    
    # visible joints
    valid_coords = ~(np.isnan(keypoints_3d).any(axis=1) | np.isinf(keypoints_3d).any(axis=1))
    if scores is not None:
        visible_joints = (scores >= confidence_threshold) & valid_coords
    else:
        visible_joints = valid_coords
    
    # h36m colors
    for i, (x, y, z) in enumerate(keypoints_3d):
        if i < len(visible_joints) and visible_joints[i] and i in keypoint_info:
            kpt_color = keypoint_info[i]["color"]
            color = [c/255.0 for c in kpt_color]  # from [0,255] to [0,1]
            
            ax.scatter(x, y, z, c=[color], s=point_size, alpha=0.9, linewidth=1)
    
    for info in skeleton_info.values():
        start_name, end_name = info["link"]
        start_id = next(k for k, v in keypoint_info.items() if v["name"] == start_name)
        end_id = next(k for k, v in keypoint_info.items() if v["name"] == end_name)
        
        if (start_id < len(visible_joints) and end_id < len(visible_joints) and
            visible_joints[start_id] and visible_joints[end_id]):
            
            start_point = keypoints_3d[start_id]
            end_point = keypoints_3d[end_id]
            
            # reasonable connection distance (avoid artifacts)
            distance = np.linalg.norm(end_point - start_point)
            if distance < 2.0:  # reasonable human body proportion
                line_color = [c/255.0 for c in info["color"]]  # [0,1] range
                
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       color=line_color, linewidth=line_width, alpha=0.8)


def create_side_by_side_2d_3d_frame(original_frame, keypoints_2d, scores_2d, keypoints_3d, 
                                    frame_title="3D Human Pose", frame_width=640, frame_height=480, fixed_limits=None):
    """
    Create a side-by-side visualization with 2D keypoints on left and 3D keypoints on right.
    
    Args:
        original_frame: np.ndarray, original video frame
        keypoints_2d: np.ndarray of shape (17, 2) with 2D keypoint coordinates
        scores_2d: np.ndarray of shape (17,) with 2D confidence scores
        keypoints_3d: np.ndarray of shape (17, 3) with 3D keypoint coordinates
        frame_title: str, title for the 3D plot
        frame_width: int, width of each side
        frame_height: int, height of the frame
        fixed_limits: np.array of shape (3, 2) with min and max values for each axis for plotting 3d skeleton
        
    Returns:
        np.ndarray: BGR image array suitable for video writing
    """
    total_width = frame_width * 2
    total_height = frame_height
    
    dpi = 100
    fig_width = total_width / dpi
    fig_height = total_height / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black')
    
    # 2d plot
    ax1 = fig.add_subplot(121)
    
    frame_with_2d = original_frame.copy()
    if keypoints_2d is not None and len(keypoints_2d) > 0:
        frame_with_2d = draw_skeleton(frame_with_2d, keypoints_2d[None, ...], scores_2d[None, ...], radius=4, line_width=2)
    
    ax1.imshow(cv2.cvtColor(frame_with_2d, cv2.COLOR_BGR2RGB))
    ax1.set_title('2D Keypoints', color='white', fontsize=12)
    ax1.axis('off')
    
    # 3d plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor('w')
    ax2.yaxis.pane.set_edgecolor('w')
    ax2.zaxis.pane.set_edgecolor('w')
    ax2.grid(False)
    
    if keypoints_3d is not None and len(keypoints_3d) > 0:
        draw_3d_skeleton_improved(
            ax=ax2,
            keypoints_3d=keypoints_3d,
            scores=scores_2d,  # 2D scores for 3D visualization # TODO: CHECK IF BETTER TO USE 3D SCORES
            confidence_threshold=0, # show all joints #0.3,
            point_size=80,
            line_width=3,
            fixed_limits=fixed_limits,
        )
    else:
        ax2.text(0, 0, 0, "No 3D keypoints detected", 
               horizontalalignment='center',
               verticalalignment='center',
               color='white', fontsize=12)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([-1, 1])
    
    ax2.set_title(frame_title, color='white', fontsize=12)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    
    img_rgb = img_array[:, :, :3] # RGBA to RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # RGB to BGR
    
    plt.close(fig)
    
    img_bgr = cv2.resize(img_bgr, (total_width, total_height))
    
    return img_bgr


def setup_3d_plot(ax, keypoints_3d=None, fixed_limits=True):
    """
    Wrapper function for backwards compatibility. Uses the improved version by default.
    
    Args:
        ax: matplotlib 3D axes object
        keypoints_3d: np.ndarray of shape (17, 3) with 3D keypoint coordinates
        fixed_limits: bool, if True uses smaller padding for more stable view
    """
    padding_factor = 0.1 if fixed_limits else 0.2
    setup_3d_plot_improved(ax, keypoints_3d, padding_factor)


def draw_3d_skeleton_matplotlib(ax, keypoints_3d, scores=None, confidence_threshold=0.3, 
                               point_size=80, line_width=3, show_joint_labels=False):
    """
    Wrapper function for backwards compatibility. Uses the improved version by default.
    
    Args:
        ax: matplotlib 3D axes object
        keypoints_3d: np.ndarray of shape (17, 3) with 3D keypoint coordinates
        scores: np.ndarray of shape (17,) with confidence scores (optional)
        confidence_threshold: float, minimum confidence to show a joint
        point_size: int, size of the points
        line_width: int, width of the skeleton lines
        show_joint_labels: bool, ignored in new version (for backwards compatibility)
    """
    draw_3d_skeleton_improved(ax, keypoints_3d, scores, confidence_threshold, 
                             max(point_size, 100), line_width)



def save_annotated_line_plots_improved(result: Dict, output_folder: str, confidence_threshold: float, 
                                     plot_confidence: bool = True, is_filtered: bool = False, 
                                     filtering_window_size: int = 0, filtering_method: str = "soft", minimum_frames_for_prediction: int = 0, added_text: str = ""):
    """Save improved annotated line plots with better visualization and interactivity.
    
    Creates a visualization showing:
    - Ground truth as continuous lines (3x thicker than predictions)
    - Predicted positions as lines with hover info
    - Closer spacing between categories
    - Interactive hover showing frame number and class
    - Optional confidence plot on secondary y-axis
    """
    
    frame_predictions_key = "all_predictions" if not is_filtered else "filtered_predictions"
    frame_confidences_key = "all_confidences" if not is_filtered else "filtered_confidences"
    
    unique_classes = sorted(set(result[frame_predictions_key] + result["all_gt_annotations"]))
    if "No Detection" in unique_classes:
        unique_classes.remove("No Detection")
    
    category_spacing = 0.05
    y_positions = {cls: i * category_spacing for i, cls in enumerate(unique_classes)}
    
    assert len(result[frame_predictions_key]) == len(result['all_gt_annotations'])
    assert len(result[frame_predictions_key]) == result['total_frames']
    
    frames = list(range(len(result[frame_predictions_key])))
    
    if plot_confidence and result[frame_confidences_key]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    fig.update_yaxes(
        tickmode='array',
        tickvals=list(y_positions.values()),
        ticktext=list(y_positions.keys()),
        title_text='Class',
        secondary_y=False  # SPECIFY PRIMARY Y-AXIS ONLY
    )
    
    gt_legend_added = False
    current_gt_sequence_frames, current_gt_sequence_positions, current_gt_sequence_labels = [], [], []
    current_gt = None
    
    for frame, gt in enumerate(result["all_gt_annotations"]):
        if gt != "No Detection" and gt in y_positions:
            if current_gt != gt and current_gt_sequence_frames:
                if len(current_gt_sequence_frames) > 0:
                    fig.add_trace(go.Scatter(
                        x=current_gt_sequence_frames,
                        y=current_gt_sequence_positions,
                        opacity=0.5,
                        mode='lines+markers',
                        line=dict(color='red', width=10),  # thickness
                        marker=dict(color='red', size=10),
                        name='Ground Truth',
                        showlegend=not gt_legend_added,  # show legend once
                        hovertemplate='<b>Ground Truth</b><br>' +
                                     'Frame: %{x}<br>' +
                                     'Class: %{text}<br>' +
                                     '<extra></extra>',
                        text=current_gt_sequence_labels
                    ))
                    gt_legend_added = True
                # new sequence
                current_gt_sequence_frames, current_gt_sequence_positions, current_gt_sequence_labels = [], [], []
            
            current_gt_sequence_frames.append(frame)
            current_gt_sequence_positions.append(y_positions[gt])
            current_gt_sequence_labels.append(gt)
            current_gt = gt
    
    # final GT sequence
    if current_gt_sequence_frames:
        fig.add_trace(go.Scatter(
            x=current_gt_sequence_frames,
            y=current_gt_sequence_positions,
            mode='lines+markers',
            line=dict(color='red', width=10),  
            marker=dict(color='red', size=10),
            opacity=0.5,
            name='Ground Truth',
            showlegend=not gt_legend_added,  #show legend if not added yet
            hovertemplate='<b>Ground Truth</b><br>' +
                         'Frame: %{x}<br>' +
                         'Class: %{text}<br>' +
                         '<extra></extra>',
            text=current_gt_sequence_labels
        ))
    
    # predictions as connected lines with hover info
    pred_legend_added = False
    current_sequence_frames, current_sequence_positions, current_sequence_labels = [], [], []
    current_pred = None
    
    for frame, pred in enumerate(result[frame_predictions_key]):
        if pred != "No Detection" and pred in y_positions:
            if current_pred != pred and current_sequence_frames:
                if len(current_sequence_frames) > 0:
                    fig.add_trace(go.Scatter(
                        x=current_sequence_frames,
                        y=current_sequence_positions,
                        mode='lines+markers',
                        line=dict(color='blue', width=2),
                        marker=dict(color='blue', size=4),
                        name='Predictions',
                        showlegend=not pred_legend_added,  
                        hovertemplate='<b>Prediction</b><br>' +
                                     'Frame: %{x}<br>' +
                                     'Class: %{text}<br>' +
                                     '<extra></extra>',
                        text=current_sequence_labels
                    ))
                    pred_legend_added = True
                current_sequence_frames, current_sequence_positions, current_sequence_labels = [], [], []
            
            current_sequence_frames.append(frame)
            current_sequence_positions.append(y_positions[pred])
            current_sequence_labels.append(pred)
            current_pred = pred
    
    if current_sequence_frames:
        fig.add_trace(go.Scatter(
            x=current_sequence_frames,
            y=current_sequence_positions,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(color='blue', size=4),
            name='Predictions',
            showlegend=not pred_legend_added, 
            hovertemplate='<b>Prediction</b><br>' +
                         'Frame: %{x}<br>' +
                         'Class: %{text}<br>' +
                         '<extra></extra>',
            text=current_sequence_labels
        ))
    
    if plot_confidence and result[frame_confidences_key]:
        confidence_trace = go.Scatter(
            x=frames,
            y=result[frame_confidences_key],
            mode='lines',
            line=dict(color='green', width=1, dash='dot'),
            name='Confidence',
            opacity=0.6,
            hovertemplate='<b>Confidence</b><br>' +
                         'Frame: %{x}<br>' +
                         'Confidence: %{y:.3f}<br>' +
                         '<extra></extra>',
            yaxis='y2'
        )
        
        if hasattr(fig, 'add_trace'):
            fig.add_trace(confidence_trace, secondary_y=True)
        else:
            fig.add_trace(confidence_trace)
    
    min_height = 120  # Minimum height # HACK
    max_height = 800  # Maximum height # HACK
    height_per_class = 60  # Height per class # HACK
    dynamic_height = min_height + (len(unique_classes) * height_per_class)
    dynamic_height = max(min_height, min(dynamic_height, max_height)) 
    
    title_text = f'Timeline of Predictions vs Ground Truth<br>{result["video_name"]}'
    if is_filtered and filtering_window_size > 0:
        title_text += f'<br>Filter: {filtering_method} (window size: {filtering_window_size})'
        title_text += f'<br>Minimum frames for prediction: {minimum_frames_for_prediction}'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title='Frame',
        yaxis_title='Position',
        hovermode='closest',
        legend=dict(x=1.02, y=1),
        width=1200,
        height=dynamic_height,  # DYNAMIC HEIGHT BASED ON NUMBER OF CLASSES
        # annotations=[
        #     dict(
        #         text=f'Confidence threshold: {confidence_threshold}',
        #         xref="paper", yref="paper",
        #         x=0.02, y=0.98,
        #         showarrow=False,
        #         font=dict(size=12)
        #     )
        # ]
    )
        
    if plot_confidence and result[frame_confidences_key]:
        if hasattr(fig, 'update_yaxes'):
            fig.update_yaxes(
                title_text="Confidence",
                range=[0, 1],
                tickmode='linear',  # ENSURE PROPER TICK MODE FOR CONFIDENCE
                secondary_y=True
            )
    
    # SPECIFY WHICH AXIS FOR EACH
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)', secondary_y=False)  # ONLY PRIMARY Y-AXIS
    
    output_name = f"{result['video_name']}_timeline_interactive_{'filtered' if is_filtered else 'unfiltered'}"
    if is_filtered and filtering_window_size > 0:
        output_name += f"_filtering-window-size-{filtering_window_size}"
    if added_text:
        output_name += f"_{added_text}"
    
    html_path = os.path.join(output_folder, f"{output_name}.html")
    fig.write_html(html_path)
    
    png_path = os.path.join(output_folder, f"{output_name}.png")
    fig.write_image(png_path, width=1200, height=600, scale=2)
    
    print(f"Interactive plot saved to: {html_path}")
    print(f"Static plot saved to: {png_path}")


def save_annotated_line_plots_matplotlib(result: Dict, output_folder: str, confidence_threshold: float, 
                                        plot_confidence: bool = True, is_filtered: bool = False, 
                                        filtering_window_size: int = 0, added_text: str = ""):
    """Original matplotlib version of the function (renamed for compatibility)"""
    plt.figure(figsize=(15, 8))

    frame_predictions_key = "all_predictions" if not is_filtered else "filtered_predictions"
    frame_confidences_key = "all_confidences" if not is_filtered else "filtered_confidences"
    
    
    unique_classes = sorted(set(result[frame_predictions_key] + result["all_gt_annotations"]))
    if "No Detection" in unique_classes:
        unique_classes.remove("No Detection")
    
    y_positions = {cls: i for i, cls in enumerate(unique_classes)}
    
    assert len(result[frame_predictions_key]) == len(result['all_gt_annotations'])
    assert len(result[frame_predictions_key]) == result['total_frames']
    
    frames = range(len(result[frame_predictions_key]))
    
    gt_x, gt_y = [], []
    current_gt = None
    for frame, gt in enumerate(result["all_gt_annotations"]):
        if gt != "No Detection":
            if current_gt != gt and current_gt is not None:
                gt_x.append(frame) # - 0.5)
                gt_y.append(None)
            gt_x.append(frame)
            gt_y.append(y_positions[gt])
            current_gt = gt
        else:
            gt_x.append(frame)
            gt_y.append(None)
    
    if gt_x:
        plt.plot(gt_x, gt_y, color='red', alpha=0.5, linewidth=3, 
                label='Ground Truth')
    
    pred_x, pred_y = [], []
    current_pred = None
    for frame, pred in enumerate(result[frame_predictions_key]):
        if pred != "No Detection":
            if current_pred != pred and current_pred is not None:
                pred_x.append(frame) # - 0.5)
                pred_y.append(None)
            pred_x.append(frame)
            pred_y.append(y_positions[pred])
            current_pred = pred
        else:
            pred_x.append(frame)
            pred_y.append(None)

    if pred_x:
        plt.plot(pred_x, pred_y, color='blue', alpha=0.7, linewidth=2,
                label='Predictions')
    
    plt.yticks(range(len(unique_classes)), unique_classes)
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.title(f'Timeline of Predictions vs Ground Truth\n{result["video_name"]}')
    plt.suptitle(f'Confidence threshold: {confidence_threshold}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if result[frame_confidences_key] and plot_confidence:
        ax2 = plt.gca().twinx()  
        ax2.plot(frames, result[frame_confidences_key], 'g-', alpha=0.3, label='Confidence')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()

    output_name = f"{result['video_name']}_timeline_{'filtered' if is_filtered else 'unfiltered'}"
    if is_filtered and filtering_window_size > 0:
        output_name += f"_filtering-window-size-{filtering_window_size}_not_shifted"
    if added_text:
        output_name += f"_{added_text}"
    
    plot_path = os.path.join(output_folder, f"{output_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()



def sliding_window_filtering(
    config: Dict, 
    action_probabilities: List[List[float]], 
    predictions: List[str], 
    filtering_window_size: int, 
    minimum_frames_for_prediction: int, 
    filtering_method: str="soft", 
    gaussian_sigma_factor: float=2.5,
    all_gt_annotations: List[str]=None,
    ) -> str:
    """
    Filtering the predictions using a sliding window. 
    Using the biggest probability of the predictions in the window. 
    In case there are too little predictions, we return "No Detection".

    Args:
        config: Dict, config dictionary
        action_probabilities: List[List[float]], action probabilities for each frame
        predictions: List[str], predictions for each frame
        filtering_window_size: int, size of the sliding window
        minimum_frames_for_prediction: int, minimum number of frames for a prediction
        filtering_method: str, method of filtering, "soft" or "weighted_gaussian"
        gaussian_sigma_factor: float, factor to adjust the Gaussian standard deviation. Use 2 for a wider spread (more uniform weights), 3 for a narrower spread (more concentrated weights), 2.5 is a good middle ground.

    Returns:
        prediction: str, filtered prediction
        confidence: float, confidence of the filtered prediction
    """
    assert len(action_probabilities) == len(predictions)
    assert len(action_probabilities[0]) == len(config["label_dict"])

    prediction = "No Detection"
    confidence = 0.0
    filtered_predictions = []
    filtered_confidences = []

    reverse_label_dict = {v: k for k, v in config["label_dict"].items()}
    
    
    # TODO: THIS IS A PLACEHOLDER FUNCTION, NOT CORRECTLY IMPLEMENTED
    if config["labeled_videos"]:
        relevant_predictions_mask = np.where(np.array(predictions) != "No Detection")
        relevant_predictions = list(np.array(predictions)[relevant_predictions_mask])
        relevant_action_probabilities = np.array(action_probabilities)[relevant_predictions_mask]

        prediction = max(set(relevant_predictions), key=relevant_predictions.count) if relevant_predictions else "No Detection"
        
        # GET CONFIDENCES FOR THE PREDICTION FROM THE ACTION PROBABILITIES
        if prediction != "No Detection":
            prediction_index = config["label_dict"][prediction]
            prediction_confidences = relevant_action_probabilities[:, prediction_index]
            confidence = np.mean(prediction_confidences) if prediction_confidences.size > 0 else 0.0
        else:
            confidence = 0.0
    
    else:
        num_frames = len(predictions)
        n_actions = len(config["label_dict"])

        filtered_predictions = []
        filtered_confidences = []

        for i in range(num_frames):
            # TODO: remove gt from here just for testing
            #gt = all_gt_annotations[i]
            #gt_index = config["label_dict"][gt]

            start_idx = max(0, i - filtering_window_size)
            end_idx = min(num_frames, i + filtering_window_size + 1)
            
            current_window_size = end_idx - start_idx
            window_action_probabilities = np.array(action_probabilities[start_idx:end_idx])

            if filtering_method == "weighted_gaussian":
                sigma = filtering_window_size / gaussian_sigma_factor  
                x_values = np.arange(-filtering_window_size, filtering_window_size + 1)
                weights = np.exp(-0.5 * (x_values / sigma) ** 2)
                weights /= np.sum(weights)  # normalize the weights

                current_weights = weights[(filtering_window_size - (i - start_idx)):(filtering_window_size + (end_idx - i))]
                current_weights = current_weights[:current_window_size]  
                window_action_probabilities *= current_weights[:, np.newaxis]
            
            window_action_probabilities_sum = np.sum(window_action_probabilities, axis=0)
            
            if np.all(window_action_probabilities_sum == 0):
                prediction = "No Detection"
                confidence = 0.0
            else:
                prediction_index = np.argmax(window_action_probabilities_sum)
                prediction = reverse_label_dict[prediction_index]
                if filtering_method == "weighted_gaussian":
                    confidence = window_action_probabilities_sum[prediction_index] / np.sum(current_weights)   # this is mean confidence of the window
                    #confidence = action_probabilities[i][prediction_index]                                     # this is the confidence of the current frame prediction
                    #confidence = action_probabilities[i][gt_index]                                              # this is the confidence of the current frame ground truth
                else:
                    confidence = window_action_probabilities_sum[prediction_index] / n_actions
            
            filtered_predictions.append(prediction)
            filtered_confidences.append(confidence)
        
        if filtered_predictions:
            filtered_predictions = apply_minimum_frames_filter(filtered_predictions, minimum_frames_for_prediction)
            
            if len(filtered_predictions) > 0:
                prediction = max(set(filtered_predictions), key=filtered_predictions.count)
                prediction_index = config["label_dict"].get(prediction, -1)
                if prediction_index != -1:
                    confidence = np.mean([conf for pred, conf in zip(filtered_predictions, filtered_confidences) if pred == prediction])
                else:
                    confidence = 0.0
            else:
                prediction = "No Detection"
                confidence = 0.0
        else:
            prediction = "No Detection" 
            confidence = 0.0

    return prediction, confidence, filtered_predictions, filtered_confidences

def apply_minimum_frames_filter(predictions: List[str], minimum_frames: int) -> List[str]:
    """
    Filter predictions by replacing sequences shorter than minimum_frames with surrounding class.
    
    Args:
        predictions: List of prediction strings
        minimum_frames: Minimum consecutive frames required for a prediction to be valid
        
    Returns:
        Filtered list of predictions
    """
    if not predictions or minimum_frames <= 1:
        return predictions
    
    i = 0
    while i < len(predictions):
        current_pred = predictions[i]
        sequence_start = i
        
        while i < len(predictions) and predictions[i] == current_pred:
            i += 1
        sequence_end = i - 1
        sequence_length = sequence_end - sequence_start + 1
        
        if sequence_length < minimum_frames:
            surrounding_class = get_surrounding_class(predictions, sequence_start, sequence_end)
            predictions[sequence_start:sequence_end + 1] = [surrounding_class] * (sequence_end - sequence_start + 1)
    
    return predictions

def get_surrounding_class(predictions: List[str], start: int, end: int) -> str:
    """
    Determine the surrounding class for a sequence that needs to be replaced.
    
    Args:
        predictions: List of predictions
        start: Start index of sequence to replace
        end: End index of sequence to replace
        
    Returns:
        Class name to use as replacement
    """
    before_class = predictions[start - 1] if start > 0 else None
    after_class = predictions[end + 1] if end < len(predictions) - 1 else None
    
    if before_class and after_class and before_class == after_class:
        return before_class
    
    if before_class and not after_class:
        return before_class
    if after_class and not before_class:
        return after_class
        
    if before_class and after_class:
        if before_class != "No Detection":
            return before_class
        return after_class
    
    return "No Detection"



