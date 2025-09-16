import time
from contextlib import contextmanager

import numpy as np
import random
import torch
import pytorch_lightning as L

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y

def blazepose33tococo17(x):
    """
    BlazePose33 keypoints mapping to COCO17
    
    Input: x (M x T x V x C) where V=33 (BlazePose33 keypoints)
    Output: y (M x T x 17 x C) (COCO17 keypoints)
        
    Blazepose33: 
        0. Nose
        1. Left eye inner
        2. Left eye
        3. Left eye outer
        4. Right eye inner
        5. Right eye
        6. Right eye outer
        7. Left ear
        8. Right ear
        9. Mouth left
        10. Mouth right
        11. Left shoulder
        12. Right shoulder
        13. Left elbow
        14. Right elbow
        15. Left wrist
        16. Right wrist
        17. Left pinky #1 knuckle
        18. Right pinky #1 knuckle
        19. Left index #1 knuckle
        20. Right index #1 knuckle
        21. Left thumb #2 knuckle
        22. Right thumb #2 knuckle
        23. Left hip
        24. Right hip
        25. Left knee
        26. Right knee
        27. Left ankle
        28. Right ankle
        29. Left heel
        30. Right heel
        31. Left foot index
        32. Right foot index
        
    COCO17:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    # init output array with shape (M, T, 17, C)
    y = np.zeros((*x.shape[:-2], 17, x.shape[-1]))
    
    # direct mappings
    y[:, :, 0, :] = x[:, :, 0, :]   # nose -> nose
    y[:, :, 1, :] = x[:, :, 2, :]   # left_eye -> left_eye  
    y[:, :, 2, :] = x[:, :, 5, :]   # right_eye -> right_eye
    y[:, :, 3, :] = x[:, :, 7, :]   # left_ear -> left_ear
    y[:, :, 4, :] = x[:, :, 8, :]   # right_ear -> right_ear
    y[:, :, 5, :] = x[:, :, 11, :]  # left_shoulder -> left_shoulder
    y[:, :, 6, :] = x[:, :, 12, :]  # right_shoulder -> right_shoulder
    y[:, :, 7, :] = x[:, :, 13, :]  # left_elbow -> left_elbow
    y[:, :, 8, :] = x[:, :, 14, :]  # right_elbow -> right_elbow
    y[:, :, 9, :] = x[:, :, 15, :]  # left_wrist -> left_wrist
    y[:, :, 10, :] = x[:, :, 16, :] # right_wrist -> right_wrist
    y[:, :, 11, :] = x[:, :, 23, :] # left_hip -> left_hip
    y[:, :, 12, :] = x[:, :, 24, :] # right_hip -> right_hip
    y[:, :, 13, :] = x[:, :, 25, :] # left_knee -> left_knee
    y[:, :, 14, :] = x[:, :, 26, :] # right_knee -> right_knee
    y[:, :, 15, :] = x[:, :, 27, :] # left_ankle -> left_ankle
    y[:, :, 16, :] = x[:, :, 28, :] # right_ankle -> right_ankle
    
    return y


KEYPOINT_MAPPINGS = {
    'coco17': {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    },
    'h36m': {
        'root': 0, 'right_hip': 1, 'right_knee': 2, 'right_foot': 3,
        'left_hip': 4, 'left_knee': 5, 'left_foot': 6, 'spine': 7,
        'thorax': 8, 'neck_base': 9, 'head': 10, 'left_shoulder': 11,
        'left_elbow': 12, 'left_wrist': 13, 'right_shoulder': 14,
        'right_elbow': 15, 'right_wrist': 16
    },
    'blazepose33': {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7,
        'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
        'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15,
        'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
        'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
        'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27,
        'right_ankle': 28, 'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
        'right_foot_index': 32
    }
}




def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextmanager
def timer(name: str):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{name} took {elapsed_time:.4f} seconds")


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds (function: {func.__name__})')
        return result
    return wrapper