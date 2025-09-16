import logging
import time

import cv2
import onnxruntime
import numpy as np
from abc import ABC, abstractmethod
import warnings
import mediapipe as mp

from ultralytics import YOLO

class KeypointModel(ABC):
    @abstractmethod
    def preprocess(self, img):
        pass

    @abstractmethod
    def __call__(self, img):
        """
        This is the main function that runs the model on the input image.
        It should return the outputs of the model, which are 
        """
        pass

    @abstractmethod
    def postprocess(self, outputs, ratio=1.0):
        pass

class RTMO(KeypointModel):

    def __init__(
        self,
        model_ckpt: str,
        input_size: tuple[int, int] = (640, 640),
        nms_sum_distance_threshold: float = 100.0,
        nms_per_keypoint_dist_threshold: float = 10.0,
        nms_num_keypoints_under_threshold: int = 8,
        provider_priority: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        available_providers = set(onnxruntime.get_available_providers())
        providers = [provider for provider in provider_priority if provider in available_providers]
        self.model = onnxruntime.InferenceSession(model_ckpt, providers=providers)
        self.input_size = input_size
        self.nms_sum_distance_threshold = nms_sum_distance_threshold
        self.nms_per_keypoint_dist_threshold = nms_per_keypoint_dist_threshold
        self.nms_num_keypoints_under_threshold = nms_num_keypoints_under_threshold

        self._startup()

    def _startup(self):
        sess_input = {
            self.model.get_inputs()[0]
            .name: np.random.rand(1, 3, self.input_size[0], self.input_size[1])
            .astype(np.float32)
        }
        sess_output = []
        for out in self.model.get_outputs():
            sess_output.append(out.name)

        _ = self.model.run(sess_output, sess_input)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114

        ratio = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[: padded_shape[0], : padded_shape[1]] = resized_img

        return padded_img, ratio
    def postprocess(
        self,
        outputs: list[np.ndarray],
        ratio: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:

        """Do postprocessing for RTMO model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMO model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        bbox_outputs, pose_outputs = outputs

        threshold_indexes = [i for i in bbox_outputs[0, :, 4] > 0.5]

        keypoints, scores = pose_outputs[0, :, :, :2], pose_outputs[0, :, :, 2]
        keypoints = keypoints / ratio

        keypoints = keypoints[threshold_indexes]
        scores = scores[threshold_indexes]

        # NMS
        dx = keypoints[:, np.newaxis, :, 0] - keypoints[np.newaxis, :, :, 0]
        dy = keypoints[:, np.newaxis, :, 1] - keypoints[np.newaxis, :, :, 1]
        pairwise_keypoint_distances = np.hypot(dx, dy)
        pairwise_distances = pairwise_keypoint_distances.sum(axis=-1)
        avg_score = scores.mean(axis=1)

        keep_indexes = []
        evaluated_indexes = set()

        for i in range(pairwise_distances.shape[0]):
            if i not in evaluated_indexes:
                sum_duplicates_flags = pairwise_distances[i] < self.nms_sum_distance_threshold
                per_keypoint_duplicates_flags = (
                    pairwise_keypoint_distances[i] < self.nms_per_keypoint_dist_threshold
                ).sum(axis=-1) > self.nms_num_keypoints_under_threshold
                duplicates = np.where(sum_duplicates_flags | per_keypoint_duplicates_flags)[0]
                evaluated_indexes.update(duplicates)
                keep_indexes.append(duplicates[np.argmax(avg_score[duplicates])])

        return keypoints[keep_indexes], scores[keep_indexes]
    def __call__(self, img: np.ndarray) -> list[np.ndarray]:
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        sess_input = {self.model.get_inputs()[0].name: input}
        sess_output = []
        for out in self.model.get_outputs():
            sess_output.append(out.name)

        outputs = self.model.run(sess_output, sess_input)

        return outputs


class YOLO11(KeypointModel):
    def __init__(self, model_ckpt, input_size=(640, 640), provider_priority=["CUDAExecutionProvider", "CPUExecutionProvider"], verbose=False):
        available_providers = set(onnxruntime.get_available_providers())
        self.providers = [p for p in provider_priority if p in available_providers]
        #self.model = onnxruntime.InferenceSession(model_ckpt, providers=providers)
        self.input_size = input_size
        self.model = YOLO(model_ckpt)

    def preprocess(self, img):
        return img, 1.0

    def __call__(self, img):
        #input_name = self.model.get_inputs()[0].name
        #outputs = self.model.run(None, {input_name: input_tensor})
        outputs = self.model(img,
            conf=0.25,
            iou=0.45,
            classes=[0],
            agnostic_nms=True,
            visualize=False,
            imgsz=self.input_size,
            device=0,
        )
        return outputs

    def postprocess(self, outputs, ratio=1.0):
        all_keypoints = []
        all_scores = []

        if len(outputs) > 1:
            raise ValueError("YOLO11 model should return only one result")

        for result in outputs:
            xy = result.keypoints.xy.detach().cpu().numpy()  # x and y coordinates
            xyn = result.keypoints.xyn.detach().cpu().numpy()  # normalized
            kpts = result.keypoints.data.detach().cpu().numpy()  # x, y, visibility (if available)
            conf = result.keypoints.conf.detach().cpu().numpy()

        return np.array(xy), np.array(conf)

class BlazePose(KeypointModel):
    """MediaPipe BlazePose model implementation for pose estimation.
    
    MediaPipe Pose detects 33 landmarks on a human body and outputs 3D coordinates
    with visibility scores. This implementation converts to 2D keypoints compatible
    with the existing pipeline.
    """
    
    def __init__(
        self,
        model_ckpt: str = None,  # for compatibility
        input_size: tuple[int, int] = (640, 640),  # for compatibility
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        static_image_mode: bool = True,  # True for video processing frame by frame
        is_3d: bool = False,
    ):
        self.input_size = input_size
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode

        self.is_3d = is_3d
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=1,  # 0, 1, or 2, higher is more accurate but slower
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self._current_frame_shape = None # for coordinate conversion

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Preprocess image for MediaPipe BlazePose.
        
        MediaPipe expects RGB images and handles resizing internally.
        We just need to ensure the image is in RGB format and store frame shape.
        
        Args:
            img (np.ndarray): Input image in RGB format.
            
        Returns:
            tuple: (processed_img, ratio) where ratio is always 1.0 for MediaPipe
        """
        self._current_frame_shape = img.shape[:2]  # (height, width)
        #processed_img = img.copy() # TODO: check if this is needed
        ratio = 1.0
        
        return img, ratio
    
    def __call__(self, img: np.ndarray) -> object:
        """Run MediaPipe BlazePose inference on the image.
        
        Args:
            img (np.ndarray): Preprocessed image in RGB format.
            
        Returns:
            MediaPipe pose results object.
        """
        results = self.pose.process(img)
        return results
    
    def postprocess(self, outputs, ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Convert MediaPipe results to keypoints and scores format.
        
        Args:
            outputs: MediaPipe pose results object.
            ratio (float): Not used for MediaPipe since it handles scaling internally.
            
        Returns:
            tuple: (keypoints, scores) where:
                - keypoints: np.ndarray of shape (num_people, num_landmarks, 2) with (x, y) coordinates in pixels
                - scores: np.ndarray of shape (num_people, num_landmarks) with visibility scores
        """
        if outputs.pose_landmarks is None:
            # if self.is_3d:
            #     return np.array([]), np.array([]), np.array([]) # TODO: check if good here
            # else:
            return np.array([]), np.array([])
        
        if self._current_frame_shape is None:
            raise ValueError("Frame shape not set. Make sure preprocess() was called before postprocess().")
        
        height, width = self._current_frame_shape
        landmarks = outputs.pose_landmarks.landmark
        
        keypoints = []
        scores = []
        
        for landmark in landmarks:
            # converting from normalized to pixel coordinates
            x_pixel = landmark.x * width
            y_pixel = landmark.y * height
            visibility = landmark.visibility  # (0-1)
            if self.is_3d:
                z_depth = landmark.z # TODO: check if good here?
                keypoints.append([x_pixel, y_pixel, z_depth])
            else:
                keypoints.append([x_pixel, y_pixel])
            scores.append(visibility)
        
        # detects single person, so we add person dimension
        # reshape to (num_people, num_landmarks, 2), with 2 being (x,y)
        keypoints = np.array(keypoints, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        if self.is_3d:
            keypoints = keypoints.reshape(1, -1, 3) 
        else:
            keypoints = keypoints.reshape(1, -1, 2) # -1 means the num_landmarks is inferred from the data (33 for blazepose)
        scores = scores.reshape(1, -1) 
        
        return keypoints, scores

MODEL_CLASSES = {
    "RTMO": RTMO,
    "YOLO11": YOLO11,
    "BlazePose": BlazePose,
}