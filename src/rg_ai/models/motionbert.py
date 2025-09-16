import logging
from collections import OrderedDict
from functools import partial
from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

from rg_ai.models.DSTFormer import DSTformer
from rg_ai.utils.utils_misc import coco2h36m, blazepose33tococo17

class MotionBERTLite:
    def __init__(self, model_ckpt: str, device: str = "cuda", deroot: bool = True, normalize_derooting: bool = True, history_len: int = 243, input_keypoints_format: str = "coco17", lite: bool = True):
        if lite:    
            self.model = DSTformer(
                dim_in=3,
                dim_out=3,
                dim_feat=256,
                dim_rep=512,
                depth=5,
                num_heads=8,
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                maxlen=243,
                num_joints=17,
            )
        else:
            self.model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=243,
            num_joints=17,
        )

        ckpts = torch.load(model_ckpt)["model_pos"]
        ckpts_no_module = OrderedDict()
        for k, v in ckpts.items():
            name = k[7:]
            ckpts_no_module[name] = v
        self.model.load_state_dict(ckpts_no_module, strict=True)
        self.model.eval()
        self.model = self.model.to(device)
        # self.model.compile()
        self.history = []
        self.history_len = history_len # 243 because learned on 243 frames, but we can pass less keypoints in the model
        self.deroot = deroot
        self.normalize_derooting = normalize_derooting
        self.input_keypoints_format = input_keypoints_format
        logging.log(logging.INFO, f"MotionBert Model compiled")

    def preprocess_keypoints(self, keypoints, camera, visibility_threshold=0.9, input_keypoints_format="coco17"):
        # [n_frames, 17, 3]
        """
        Args:
            keypoints: np.ndarray, shape: [n_frames, n_keypoints, 3]
            camera: tuple, shape: (width, height)
            visibility_threshold: float, hack to put 0, since it works better than true visibility
            input_keypoints_format: str, "coco17" or "h36m" or "blazepose33"

        Returns:
            keypoints: np.ndarray, shape: [1, n_frames, 17 (coco17), 3] (for one person)
        """
        w, h = camera
        ### HACK - FIXED VISIBILITY TO 1 ###
        keypoints[..., 2] = np.where(keypoints[..., 2] > visibility_threshold, 1.0, 0.0)
        # DEROOTING 
        if self.deroot:
            keypoints[..., :2] -= keypoints[..., :1, :2]
            if self.normalize_derooting:
                keypoints[..., :2] *= h * 0.5 / self.compute_bbox_diagonals(keypoints) # TODO: MAYBE CHANGE TO ANOTHER SCALING
                keypoints[..., :2] += [w/2, h/2]
        # CONCATENATE VISIBILITY
        persons_keypoints = [keypoints] # [1, n_frames, 17, 3]
        keypoints = np.stack(
            np.stack([person_keypoints[-self.history_len :] for person_keypoints in persons_keypoints], axis=0), axis=0
        )
        keypoints[..., :2] = keypoints[..., :2] / w * 2 - [1, h / w] # normalize based on camera resolution
        if self.input_keypoints_format == "coco17":
            return coco2h36m(keypoints).astype(np.float32) 

        elif self.input_keypoints_format == "blazepose33": 
            return coco2h36m(blazepose33tococo17(keypoints)).astype(np.float32)
    
    def compute_bbox_diagonals(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute the diagonal of the bounding box of the keypoints.

        Args:
            keypoints: np.ndarray of shape [n_frames, 17, 3]

        Returns:
            mean of the diagonals of the bounding boxes of the keypoints
        """
        diagonals = np.linalg.norm(keypoints[..., :2].max(axis=1) - keypoints[..., :2].min(axis=1), axis=1)
        return diagonals.mean(axis=0) # TODO - maybe use max instead so always under 1
    
    def postprocess(self, keypoints: np.ndarray, get_frame_keypoints: str = "last") -> np.ndarray:
        '''
        Parameters:
            keypoints: keypoints, shape: [n_people, n_frames, n_joints, 3]
            get_frame_keypoints: str, "last", "middle" or "first"
            # TODO: CHECK SHAPE OF KEYPOINTS HERE! IINK THERE SHOULD BE 3D KEYPOINTS - DIFFERENT OT THE ABOVE PREPROCESSS    
        '''
        # keypoints = keypoints[..., [0, 2, 1]] # change x,z,y to x,y,z # for n_frames, shape: [n_people, n_frames, n_joints, 3]
        # keypoints = keypoints.mean(axis=1)         # mean across frames - didnt work
        # get the middle frame
        if get_frame_keypoints == "middle":
            frame_idx = keypoints.shape[1] // 2
        elif get_frame_keypoints == "last":
            frame_idx = -1
        elif get_frame_keypoints == "first":
            frame_idx = 0
        elif get_frame_keypoints == "all":
            frame_idx = None # TODO: check if needed
        # elif get_frame_keypoints == "mean_of_three":
        #     frame_idx = keypoints.shape[1] // 2
        #     # get the mean of the three frames around the middle frame
        #     keypoints = keypoints[:, frame_idx-1:frame_idx+2, :, :].mean(axis=1)
        #     frame_idx = 0
            
        else:
            raise ValueError(f"Invalid value for get_frame_keypoints: {get_frame_keypoints}")
        
        # to get - a - min(a) you can also do -(a + min(a))
        keypoints = keypoints[:, frame_idx, :, :][:, :, [0, 2, 1]] # for last frame, shape: [n_people, n_joints, 3]
        keypoints[..., 0] = -keypoints[..., 0] # negate x 
        keypoints[..., 2] = -keypoints[..., 2] # negate z

        # If rebase_keypoint_height is True, adjust z-axis values
        # TODO: check how it works without this: tested and it has z values outside 
        keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)
        return keypoints
    
    def get_3d_keypoints(self, keypoints: np.ndarray, postprocess: bool = True, get_frame_keypoints: str = "last") -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            keypoints: np.ndarray, shape: [n_people, n_frames, n_joints, 3]
            postprocess: bool, whether to postprocess the keypoints
            get_frame_keypoints: str, "last", "middle" or "first"
        """
        representation, keypoints_3d = self.model(keypoints, return_rep=False)
        # turn into numpy arrays 
        # representation = representation.cpu().numpy()
        keypoints_3d = keypoints_3d.cpu().numpy()
        
        if postprocess:
            keypoints_3d = self.postprocess(keypoints_3d, get_frame_keypoints)

        return representation, keypoints_3d

    @torch.inference_mode()
    def __call__(self, keypoints: np.ndarray) -> List[np.ndarray]:
        return self.model(keypoints)