import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2
from tqdm import tqdm
import imageio
from rg_ai.utils.plotting import draw_skeleton
from rg_ai.utils.skeleton import h36m, coco17


def get_img_from_fig(fig, dpi=120):
    """Convert matplotlib figure to numpy array"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # COLOR_BGR2RGBA
    return img

def motion2video_3d_motionbert(motion, save_path=None, fps=25, return_frames=False):
    """
    MotionBERT's optimized 3D visualization
    Args:
        motion: (17,3,N) or (N,17,3) array of 3D keypoints
        save_path: path to save video (optional)
        fps: frames per second
        return_frames: if True, return frame array instead of saving
    """
    # Handle input format - convert to (17,3,N)
    if motion.shape[0] != 17:
        motion = np.transpose(motion, (1,2,0))  # (N,17,3) -> (17,3,N)
    
    vlen = motion.shape[-1]
    frames = []

    name_to_id = {value["name"]: id_ for id_,value in h36m["keypoint_info"].items()}
    pairs = [link["link"] for link in h36m["skeleton_info"].values()]
    joint_pairs = [[name_to_id[a],name_to_id[b]] for a,b in pairs]
    
    # H36M joint pairs - MotionBERT's proven connections
    # joint_pairs = [
    #     [0, 1], [1, 2], [2, 3],     # right leg
    #     [0, 4], [4, 5], [5, 6],     # left leg  
    #     [0, 7], [7, 8], [8, 9],     # spine to head
    #     [8, 11], [8, 14],           # shoulders
    #     [9, 10],                    # head
    #     [11, 12], [12, 13],         # left arm
    #     [14, 15], [15, 16]          # right arm
    # ]
    
    # MotionBERT's color scheme
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"     # Blue for center
    color_left = "#02315E"    # Dark blue for left
    color_right = "#2F70AF"   # Light blue for right
    
    if save_path:
        videowriter = imageio.get_writer(save_path, fps=fps)
    
    for f in range(vlen): #tqdm(, desc="Rendering 3D poses"):
        j3d = motion[:, :, f]  # (17, 3)
        
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        
        # MotionBERT's optimized axis limits and viewing angle
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256) 
        ax.set_zlim(-512, 0)
        ax.view_init(elev=12., azim=80)
        
        # Clean plot - no ticks/labels
        plt.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        
        # Draw skeleton with MotionBERT's color scheme
        for i, limb in enumerate(joint_pairs):
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            
            # Choose color based on body part
            if joint_pairs[i] in joint_pairs_left:
                color = color_left
            elif joint_pairs[i] in joint_pairs_right:
                color = color_right
            else:
                color = color_mid
                
            # MotionBERT's axis transformation for better visualization
            ax.plot(-xs, -zs, -ys, color=color, lw=3, marker='o', 
                   markerfacecolor='w', markersize=3, markeredgewidth=2)
        
        frame_vis = get_img_from_fig(fig)
        if return_frames:
            frames.append(frame_vis)
        elif save_path:
            videowriter.append_data(frame_vis)
            
        plt.close()
    
    if save_path:
        videowriter.close()
        
    return frames if return_frames else None

def pixel2world_vis_motion(motion, dim=3):
    """MotionBERT's coordinate transformation for better visualization"""
    N = motion.shape[-1]
    if dim == 2:
        offset = np.ones([2, N]).astype(np.float32)
    else:
        offset = np.ones([3, N]).astype(np.float32)
        offset[2, :] = 0
    return (motion + offset) * 512 / 2

def draw_3d_skeleton_motionbert_style(ax, keypoints_3d, frame_title="3D Pose"):
    """
    Use MotionBERT's 3D visualization approach
    """
    if keypoints_3d is None or len(keypoints_3d) == 0:
        # Create a black background and center the text
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('black')
        ax.text(0.5, 0.5, "No 3D keypoints detected", 
               horizontalalignment='center', verticalalignment='center',
               color='white', fontsize=12, transform=ax.transAxes)
        ax.set_title(frame_title, color='white', fontsize=12)
        ax.axis('off')
        return
    
    # Convert to MotionBERT format and render single frame
    motion = keypoints_3d.T.reshape(17, 3, 1)  # (17, 3, 1)
    motion_world = pixel2world_vis_motion(motion, dim=3)
    
    # Get single frame from MotionBERT renderer
    frames = motion2video_3d_motionbert(motion_world, return_frames=True)
    if frames:
        # Convert frame to display in your plot
        frame = frames[0]
        ax.clear()
        ax.imshow(cv2.cvtColor(frame[:,:,:3], cv2.COLOR_BGR2RGB))
        ax.set_title(frame_title, color='white', fontsize=12)
        ax.axis('off')

def create_side_by_side_2d_3d_frame_improved(original_frame, keypoints_2d, scores_2d, keypoints_3d, 
                                            frame_title="3D Human Pose", frame_width=640, frame_height=480):
    """
    Enhanced version using MotionBERT's 3D visualization
    """
    total_width = frame_width * 2
    total_height = frame_height
    
    dpi = 100
    fig_width = total_width / dpi
    fig_height = total_height / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black')
    
    # 2D plot
    ax1 = fig.add_subplot(121)
    frame_with_2d = original_frame.copy()
    if keypoints_2d is not None and len(keypoints_2d) > 0:
        frame_with_2d = draw_skeleton(frame_with_2d, keypoints_2d[None, ...], scores_2d[None, ...], radius=4, line_width=2)
    
    ax1.imshow(cv2.cvtColor(frame_with_2d, cv2.COLOR_BGR2RGB))
    ax1.set_title('2D Keypoints', color='white', fontsize=12)
    ax1.axis('off')
    
    # 3D plot using MotionBERT's method
    ax2 = fig.add_subplot(122)
    draw_3d_skeleton_motionbert_style(ax2, keypoints_3d, frame_title)
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    
    img_rgb = img_array[:, :, :3]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    img_bgr = cv2.resize(img_bgr, (total_width, total_height))
    return img_bgr