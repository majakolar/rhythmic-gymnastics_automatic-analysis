import os
import pickle
import sys
from datetime import datetime
from collections import OrderedDict
import warnings
import random
from typing import List, Dict
import dotenv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
#from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from torchvision.datasets import MNIST
import yaml
import pandas as pd

from rg_ai.utils.utils_misc import set_seed
from rg_ai.utils.utils_annotations import get_label_from_filename
from rg_ai.utils.label_consolidation import consolidate_labels, LabelConsolidator

def extract_video_name_from_embedding_file(filename: str) -> str:
    """
    Extract video name from embedding file name.
    
    For files like: "Balance_OG20-W-C1-101-AUS-IAKOVLEVA-Lidiia-BA-Individual_seg0_Balance_Fouetté_person_0.pkl"
    Returns: "OG20-W-C1-101-AUS-IAKOVLEVA-Lidiia-BA-Individual"
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    
    if len(parts) >= 3:
        return parts[1]
    
    return basename


def load_train_test_split(videos_folders: List[str]) -> Dict[str, List[str]]:
    """
    Load train/test split from the videos folder.
    
    Args:
        videos_folders: List of video folder paths (assumes only one folder)
        
    Returns:
        Dict with 'train' and 'test' keys containing lists of video names
    """
    if not videos_folders:
        raise ValueError("No videos_folders provided")
    
    videos_folder = videos_folders[0]
    split_file_path = os.path.join(videos_folder, 'train_test_split.yaml')
    
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Train/test split file not found: {split_file_path}")
    
    with open(split_file_path, 'r') as f:
        split_data = yaml.safe_load(f)
    
    return split_data['splits']


def filter_files_by_split(folder_path: str, video_names: List[str], split_type: str) -> List[str]:
    """
    Filter files in folder based on video names from the split.
    
    Args:
        folder_path: Path to folder containing embedding files
        video_names: List of video names for this split
        split_type: 'train', 'test', or 'val' for logging
        
    Returns:
        List of file names that belong to this split
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    split_files = []
    
    for file in all_files:
        video_name = extract_video_name_from_embedding_file(file)
        if video_name in video_names:
            split_files.append(file)
    
    print(f"{split_type.capitalize()} split: {len(split_files)} files from {len(video_names)} videos")
    return split_files

def create_sublabel_to_main_mapping(joined_label_dict, label_dict):
        """Create mapping from sublabel index to main category."""
        sublabel_to_main = {}
        for main_label, sublabels in joined_label_dict.items():
            for sublabel in sublabels:
                if sublabel == "GENERAL":
                    full_label = main_label
                else:
                    full_label = f"{main_label}_{sublabel}"
                
                if full_label in label_dict:
                    sublabel_to_main[label_dict[full_label]] = main_label
        return sublabel_to_main

class FocalLoss(nn.Module):
    """Multi-class Focal Loss implemetation"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes] (logits)
        # targets: [batch_size] (class indices)

        log_probs = F.log_softmax(inputs, dim=1)  # log probabilities
        probs = torch.exp(log_probs)              # probabilities

        # probabilities of the true class
        targets = targets.view(-1, 1)
        log_p = log_probs.gather(1, targets).view(-1)
        p = probs.gather(1, targets).view(-1)

        # Focal loss
        loss = -self.alpha * (1 - p) ** self.gamma * log_p

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLossMultilabel(nn.Module):
    """Multi-label Focal Loss implemetation"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLossMultilabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss for handling class imbalance."""
    def __init__(self, class_weights=None, reduction='mean'): #gpu
        super(WeightedCrossEntropyLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights.to(self.device)
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)

class HierarchicalCrossEntropyLoss(nn.Module):
    """Hierarchical Cross-Entropy Loss for two-level label hierarchy."""
    def __init__(self, joined_label_dict, label_dict, main_weight=1.0, sub_weight=1.0, reduction='mean', sublabel_to_main=None):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        self.main_weight = main_weight
        self.sub_weight = sub_weight
        self.reduction = reduction
        
        self.sublabel_to_main = sublabel_to_main #or create_sublabel_to_main_mapping(joined_label_dict, label_dict)
        self.main_labels = list(set(self.sublabel_to_main.values()))
        self.main_to_idx = {label: idx for idx, label in enumerate(self.main_labels)}
        
        self.main_criterion = nn.CrossEntropyLoss(reduction='none')
        self.sub_criterion = nn.CrossEntropyLoss(reduction='none')
        

    
    def forward(self, inputs, targets):
        device = inputs.device
        
        main_targets = torch.tensor([self.main_to_idx[self.sublabel_to_main[t.item()]] 
                                   for t in targets], device=device)
        
        main_logits = self._aggregate_to_main_logits(inputs)
        
        main_loss = self.main_criterion(main_logits, main_targets)
        sub_loss = self.sub_criterion(inputs, targets)
        
        total_loss = self.main_weight * main_loss + self.sub_weight * sub_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
    
    def _aggregate_to_main_logits(self, sublabel_logits):
        """Aggregate sublabel logits to main category logits."""
        batch_size = sublabel_logits.size(0)
        num_main = len(self.main_labels)
        main_logits = torch.zeros(batch_size, num_main, device=sublabel_logits.device)
        
        for main_idx, main_label in enumerate(self.main_labels):
            sublabel_indices = [idx for idx, main in self.sublabel_to_main.items() if main == main_label]
            if sublabel_indices:
                main_logits[:, main_idx] = torch.logsumexp(sublabel_logits[:, sublabel_indices], dim=1)
        
        return main_logits

def create_sublabel_dict(joined_label_dict):
    """Create a flat dictionary mapping each sublabel to a unique index."""
    consolidator = LabelConsolidator()
    return consolidator.create_sublabel_dict(joined_label_dict)

class ActionDataset(Dataset):
    def __init__(self, folder_path, config, file_list=None):
        """
        Initializes the ActionDataset.

        Args:
            folder_path (str): Path to the folder containing the data files.
            config (dict): Configuration dictionary containing all dataset parameters.
            file_list (List[str], optional): Specific list of files to use. If None, uses all .pkl files in folder.
        """
        self.folder_path = folder_path
        self.config = config
        
        # use provided file list or discover all files
        if file_list is not None:
            self.files = file_list
        else:
            self.files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    
        self.join_labels = config.get("join_labels", True)
        self.label_dict = config.get("label_dict")
        self.joined_label_dict = config.get("joined_label_dict")
        
        self.use_sublabels = config.get("use_sublabels", False)
        self.use_consolidation = config.get("use_consolidation", False)

        # apply subsampling for "Other" category if enabled
        subsample_other = config.get("subsample_other", True)
        if subsample_other:
            other_subsample_ratio = config.get("other_subsample_ratio", 0.1)
            subsample_seed = config.get("subsample_seed", 42)
            self.files = self._subsample_other_category(self.files, other_subsample_ratio, subsample_seed)
        
        self.labels = {file: get_label_from_filename(
            filename=file,
            main_label_index=0,
            use_sublabels=self.use_sublabels,
            joined_label_dict=self.joined_label_dict,
            use_consolidation=self.use_consolidation,
            other_check=True, # returns Other if other is found in the filename (thats how GT is annotated)
            consolidation_map=self.config.get("consolidation_map")
        ) for file in self.files}

        # store the number of frames for each file to be able to index correctly
        self.file_frame_indices = []
        self.all_frames = 0
        for idx, file in enumerate(self.files):
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, "rb") as f:
                data = np.array(pickle.load(f))  # data shape: (B, 1, 17, 512)
            num_frames = data.shape[0]
            self.file_frame_indices.extend([(idx, frame_idx) for frame_idx in range(num_frames)])
            self.all_frames += num_frames
        print(f"Loaded {len(self.files)} files with {self.all_frames} frames")

    
    def _subsample_other_category(self, files, subsample_ratio, seed):
        """
        Subsample files from the "Other" category to balance the dataset.
        
        Args:
            files (List[str]): List of all files
            subsample_ratio (float): Ratio of "Other" files to keep
            seed (int): Random seed for reproducibility
            
        Returns:
            List[str]: Files with "Other" category subsampled
        """
        # Separate "Other" files from the rest
        other_files = []
        non_other_files = []
        
        for file in files:
            label = get_label_from_filename(
                filename=file,
                main_label_index=0,
                use_sublabels=self.use_sublabels,
                joined_label_dict=self.joined_label_dict,
                use_consolidation=self.use_consolidation,
                other_check=True, # returns Other if other is found in the filename (thats how GT is annotated)
                consolidation_map=self.config.get("consolidation_map")
            )
            if label.lower() == "other":
                other_files.append(file)
            else:
                non_other_files.append(file)
        
        print(f"Original distribution: {len(other_files)} Other files, {len(non_other_files)} non-Other files")
        
        # Subsample "Other" files
        if other_files:
            random.seed(seed)
            n_other_to_keep = max(1, int(len(other_files) * subsample_ratio))
            subsampled_other = random.sample(other_files, n_other_to_keep)
            print(f"Subsampled Other category: keeping {len(subsampled_other)} out of {len(other_files)} files ({subsample_ratio*100:.1f}%)")
        else:
            subsampled_other = []
            print("No 'Other' files found for subsampling")
        
        # Combine subsampled "Other" with all non-"Other" files
        final_files = non_other_files + subsampled_other
        print(f"Final distribution: {len(subsampled_other)} Other files, {len(non_other_files)} non-Other files")
        
        return final_files

    def __len__(self):
        """
        Returns the total number of frames in the dataset.

        Returns:
            int: Total number of frames.
        """
        return len(self.file_frame_indices)

    def __getitem__(self, idx):
        """
        Retrieves the data and label for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: A tuple containing the embedding (torch.Tensor) and label index (torch.Tensor).
        """
        file_idx, frame_idx = self.file_frame_indices[idx]
        file_name = self.files[file_idx]
        file_path = os.path.join(self.folder_path, file_name)

        with open(file_path, "rb") as f:
            # data shape: (B, 1, 17, 512)
            data = np.array(pickle.load(f))

        # specific frame
        # frame_data shape: (1, 17, 512)
        frame_data = data[frame_idx]

        embedding = torch.tensor(frame_data, dtype=torch.float32)

        label = self.labels[file_name]
        label_idx = self.label_to_index(label)

        return embedding, torch.tensor(label_idx, dtype=torch.long)

    def label_to_index(self, label):
        """Converts a label to its corresponding index."""
        if self.join_labels:   
            for key, value in self.joined_label_dict.items():
                if label in value:
                    return self.label_dict[key]
            raise ValueError(f"Label {label} not found in joined_label_dict")
        else:
            return self.label_dict[label]            

class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0.0, dim_rep=512, num_classes=9, num_joints=17, hidden_dim=2048):
        """
        Initializes the ActionHeadClassification model.

        Args:
            dropout_ratio (float): Dropout ratio for regularization.
            dim_rep (int): Dimensionality of the representation.
            num_classes (int): Number of output classes.
            num_joints (int): Number of joints in the input data.
            hidden_dim (int): Dimensionality of the hidden layer.

        Returns:
            None
        """
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat, avg_history_dim=False):
        """
        Forward pass of the model.

        Args:
            feat (torch.Tensor): Input feature tensor of shape (B, T, J, C).
            avg_history_dim (bool): Whether to average over the history dimension.

        Returns:
            torch.Tensor: Output logits of the model.
        """
        B, T, J, C = feat.shape
        if avg_history_dim:
            feat = feat.permute(0, 2, 3, 1)  # (Batch_size, T, J, C) -> (B, J, C, T)
            feat = feat.mean(dim=-1)

        feat = self.dropout(feat)
        feat = feat.reshape(B, -1)  # (B, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat


class LitActionHeadClassification(L.LightningModule):
    def __init__(self, model, config, extract_keypoints_config=None):
        """
        Initializes the LitActionHeadClassification module.

        Args:
            model (nn.Module): The model to be trained.
            config (dict): Configuration dictionary containing all training parameters.
            extract_keypoints_config (dict, optional): Configuration from keypoints extraction step.
        """
        super(LitActionHeadClassification, self).__init__()
        self.model = model
        self.config = config
        self.extract_keypoints_config = extract_keypoints_config or {}
        
        self.lr = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 4)
        self.num_workers = config.get("num_workers", 4)
        self.shuffle = config.get("shuffle", True)
        self.drop_last = config.get("drop_last", True)
        self.join_labels = config.get("join_labels", True)
        
        self.label_dict = config.get("label_dict")
        self.joined_label_dict = config.get("joined_label_dict")
        self.confusion_matrix_folder = config.get("confusion_matrix_folder")
        self.model_folder = config.get("model_folder")
        
        # main label metrics and hierarchical loss related variables
        self.sublabel_to_main = create_sublabel_to_main_mapping(self.joined_label_dict, self.label_dict)
        self.main_labels = list(set(self.sublabel_to_main.values()))
        self.main_to_idx = {label: idx for idx, label in enumerate(self.main_labels)}
        
        self.train_folder = config.get("train_folder")
        self.val_folder = config.get("val_folder")
        self.test_folder = config.get("test_folder")

        self._setup_loss_function()

        self.wrong_preds = []

        self.val_gts = []
        self.val_preds = []
        self.test_gts = []
        self.test_preds = []
        
        self.do_bootstrap = True # HACK
        self.bootstrap_results = {}

        self.curr_time_str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # dataset file lists if using train/test split
        train_test_split = self.extract_keypoints_config.get("train_test_split", True)
        videos_folders = self.extract_keypoints_config.get("videos_folders")
        
        if not train_test_split and videos_folders:
            self._setup_datasets_with_split()
        else:
            self.train_file_list = None
            self.val_file_list = None
            self.test_file_list = None

    
    def _setup_loss_function(self):
        """Setup loss function based on configuration."""
        loss_type = self.config.get("loss_function", "focal")
        class_weights_dict = self.config.get("class_weights_dict")
        
        class_weights = None 

        if class_weights_dict is not None and self.label_dict is not None:
            class_weights = torch.zeros(len(self.label_dict))
            for class_name, weight in class_weights_dict.items():
                if class_name in self.label_dict:
                    class_weights[self.label_dict[class_name]] = weight
                else:
                    warnings.warn(f"Class '{class_name}' in class_weights_dict not found in label_dict")
        
        if loss_type.lower() == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction='mean')
            print("Using Cross-Entropy Loss")
            
        elif loss_type.lower() == "weighted_cross_entropy":
            if class_weights is None:
                raise ValueError("class_weights_dict must be provided for weighted_cross_entropy loss")
            self.loss = WeightedCrossEntropyLoss(class_weights=class_weights, reduction='mean')
            print(f"Using Weighted Cross-Entropy Loss with weights: {dict(zip(self.label_dict.keys(), class_weights.tolist()))}")
            
        elif loss_type.lower() == "focal":
            focal_alpha = self.config.get("focal_alpha", 1.0)
            focal_gamma = self.config.get("focal_gamma", 2.0)
            self.loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
            print(f"Using Focal Loss (α={focal_alpha}, γ={focal_gamma})")

        elif loss_type.lower() == "focal_multilabel":
            focal_alpha = self.config.get("focal_alpha", 1.0)
            focal_gamma = self.config.get("focal_gamma", 2.0)
            self.loss = FocalLossMultilabel(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
            print(f"Using Focal Loss Multilabel(α={focal_alpha}, γ={focal_gamma})")
            
        elif loss_type.lower() == "hierarchical":
            if not self.joined_label_dict:
                raise ValueError("joined_label_dict must be provided for hierarchical loss")
            main_weight = self.config.get("hierarchical_main_weight", 1.0)
            sub_weight = self.config.get("hierarchical_sub_weight", 1.0)
            self.loss = HierarchicalCrossEntropyLoss(
                joined_label_dict=self.joined_label_dict,
                label_dict=self.label_dict,
                main_weight=main_weight,
                sub_weight=sub_weight,
                reduction='mean',
                sublabel_to_main=self.sublabel_to_main
            )
            print(f"Using Hierarchical Cross-Entropy Loss (main_weight={main_weight}, sub_weight={sub_weight})")
            
        else:
            raise ValueError(f"Unknown loss function: {loss_type}. Choose from: cross_entropy, weighted_cross_entropy, focal, hierarchical")


    def _setup_datasets_with_split(self):
        """Setup datasets using train/test split from videos folder."""
        videos_folders = self.extract_keypoints_config.get("videos_folders")
        split_data = load_train_test_split(videos_folders)
        
        # embedding files for each split
        embeddings_folder = os.path.join(os.path.dirname(self.train_folder), "all")  # HACK # Use train_folder as the embeddings/all folder
        train_files = filter_files_by_split(embeddings_folder, split_data['train'], 'train')
        test_files = filter_files_by_split(embeddings_folder, split_data['test'], 'test')
        val_files = filter_files_by_split(embeddings_folder, split_data['val'], 'val')
        
        print(f"Final split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")
        
        self.train_folder = embeddings_folder
        self.val_folder = embeddings_folder
        self.test_folder = embeddings_folder
        self.train_file_list = train_files
        self.val_file_list = val_files
        self.test_file_list = test_files

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits from the model.
        """
        return self.model(x)

    def training_step(self, batch):
        """
        Performs a single training step.

        Args:
            batch (tuple): A batch of data containing inputs and labels.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch):
        """
        Performs a single validation step.

        Args:
            batch (tuple): A batch of data containing inputs and labels.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        
        y_cpu = y.cpu()
        y_pred_cpu = y_pred.cpu()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        # predictions for current epoch metrics
        self.current_epoch_val_gts.extend(y_cpu.numpy())
        self.current_epoch_val_preds.extend(y_pred_cpu.numpy())
        
        # if last epoch or if early stopping is triggered (for final confusion matrix)
        if self.current_epoch == self.trainer.max_epochs - 1 or self.trainer.should_stop:
            self.val_gts.extend(y_cpu.numpy())
            self.val_preds.extend(y_pred_cpu.numpy())

        return loss

    def test_step(self, batch):
        """
        Performs a single test step - only collect predictions and ground truth.

        Args:
            batch (tuple): A batch of data containing inputs and labels.

        Returns:
            None
        """
        x, y = batch
        y_hat = self.model(x)
        y_pred = torch.argmax(y_hat, dim=1)
        y_cpu = y.cpu()
        y_pred_cpu = y_pred.cpu()
        
        self.test_gts.extend(y_cpu.numpy())
        self.test_preds.extend(y_pred_cpu.numpy())

    def on_validation_epoch_start(self):
        """
        Starts collecting current epoch's validation data.
        """
        self.current_epoch_val_gts = []
        self.current_epoch_val_preds = []

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch to compute metrics on entire dataset.
        """
        if len(self.current_epoch_val_gts) == 0 or len(self.current_epoch_val_preds) == 0:
            print("No validation data collected for current epoch")
            return

        gts = np.array(self.current_epoch_val_gts)
        preds = np.array(self.current_epoch_val_preds)
        
        # all possible class labels for consistent metric calculation
        all_labels = list(range(len(self.label_dict)))
        
        # Debug: Show class distribution
        #unique_gt = np.unique(gts)
        #unique_pred = np.unique(preds)
        #print(f"Validation epoch - GT classes: {unique_gt}, Pred classes: {unique_pred}")
        #print(f"Expected labels: {all_labels}")

        acc = accuracy_score(gts, preds)
        precision = precision_score(gts, preds, average='macro', labels=all_labels, zero_division=0.0)
        recall = recall_score(gts, preds, average='macro', labels=all_labels, zero_division=0.0) # == balanced accuracy
        f1 = f1_score(gts, preds, average='macro', labels=all_labels, zero_division=0.0)
    
        self.log("val_acc", acc)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        
        main_metrics = self._compute_main_label_metrics(gts, preds)
        if main_metrics:
            self.log("val_main_acc", main_metrics['main_acc'])
            self.log("val_main_precision", main_metrics['main_precision'])
            self.log("val_main_recall", main_metrics['main_recall'])
            self.log("val_main_f1", main_metrics['main_f1'])


    def on_test_epoch_end(self):
        """
        Called at the end of test epoch to compute metrics on entire dataset.
        """
        if len(self.test_gts) == 0 or len(self.test_preds) == 0:
            print("No test data collected")
            return
        
        gts = np.array(self.test_gts)
        preds = np.array(self.test_preds)
        
        all_labels = list(range(len(self.label_dict)))
        
        unique_gt = np.unique(gts)
        unique_pred = np.unique(preds)
        print(f"Test epoch - GT classes: {unique_gt}, Pred classes: {unique_pred}")
        print(f"Expected labels: {all_labels}")
        
     
        # Bootstrap analysis for uncertainty estimation
        if self.do_bootstrap:
            try:
                from rg_ai.utils.bootstrap_utils import bootstrap_classification_results, print_bootstrap_results
                
                print("\n" + "="*60)
                print("BOOTSTRAP UNCERTAINTY ANALYSIS")
                print("="*60)
                
                # all metrics including overall
                bootstrap_results_all = bootstrap_classification_results(
                    gts=gts, 
                    preds=preds, 
                    label_dict=self.label_dict, 
                    exclude_other=False,
                    metrics=['precision', 'recall', 'f1'],
                    n_bootstrap=1000,
                    random_state=42
                )
                
                print_bootstrap_results(bootstrap_results_all, "Overall Metrics (All Classes)")
                
                # hierarchical main label bootstrap analysis
                main_metrics = self._compute_main_label_metrics(gts, preds)
                if main_metrics:
                    print("\n" + "="*60)
                    print("MAIN LABEL BOOTSTRAP ANALYSIS")
                    print("="*60)
                    
                    main_label_dict = {label: idx for idx, label in enumerate(self.main_labels)}
                    
                    bootstrap_results_main = bootstrap_classification_results(
                        gts=main_metrics['main_gts'], 
                        preds=main_metrics['main_preds'], 
                        label_dict=main_label_dict, 
                        exclude_other=False,
                        metrics=['precision', 'recall', 'f1'],
                        n_bootstrap=1000,
                        random_state=42
                    )
                    
                    print_bootstrap_results(bootstrap_results_main, "Main Label Metrics")
                    self._store_bootstrap_metrics(bootstrap_results_main, "main_labels")
                
                self._store_bootstrap_metrics(bootstrap_results_all, "all_classes")
                    
            except Exception as e:
                print(f"Warning: Bootstrap analysis failed: {e}")
                print("Continuing with standard metrics only...")
        
        else:
            acc = accuracy_score(gts, preds)
            precision = precision_score(gts, preds, average='macro', labels=all_labels, zero_division=0.0)
            recall = recall_score(gts, preds, average='macro', labels=all_labels, zero_division=0.0) # == balanced accuracy
            f1 = f1_score(gts, preds, average='macro', labels=all_labels, zero_division=0.0)
        
            self.log("test_acc", acc)
            self.log("test_precision", precision)
            self.log("test_recall", recall)
            self.log("test_f1", f1)
            
            main_metrics = self._compute_main_label_metrics(gts, preds)
            if main_metrics:
                self.log("test_main_acc", main_metrics['main_acc'])
                self.log("test_main_precision", main_metrics['main_precision'])
                self.log("test_main_recall", main_metrics['main_recall'])
                self.log("test_main_f1", main_metrics['main_f1'])
            
            print(f"Test Results (Sublabel Level):")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            
            if main_metrics:
                print(f"\nTest Results (Main Label Level):")
                print(f"  Accuracy: {main_metrics['main_acc']:.4f}")
                print(f"  Precision: {main_metrics['main_precision']:.4f}")
                print(f"  Recall: {main_metrics['main_recall']:.4f}")
                print(f"  F1: {main_metrics['main_f1']:.4f}")

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        """
        Creates the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        print("Loading dataset...")
        train_dataset = ActionDataset(
            folder_path=self.train_folder, 
            config=self.config,
            file_list=self.train_file_list
        )
        self.train_dataset = train_dataset
        trainloader_params = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "drop_last": self.drop_last,
        }
        return DataLoader(train_dataset, **trainloader_params)

    def val_dataloader(self):
        """
        Creates the validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        # config copy with different subsample_seed for validation
        val_config = self.config.copy()
        val_config["subsample_seed"] = self.config.get("subsample_seed", 42) + 1
        
        val_dataset = ActionDataset(
            folder_path=self.val_folder, 
            config=val_config,
            file_list=self.val_file_list
        )
        self.val_dataset = val_dataset

        valloader_params = {
            "batch_size": self.batch_size, 
            "shuffle": False, 
            "num_workers": self.num_workers, 
            "drop_last": self.drop_last
        }

        return DataLoader(val_dataset, **valloader_params)

    def test_dataloader(self):
        """
        Creates the test data loader.

        Returns:
            DataLoader: The test data loader.
        """
        # config copy with no subsampling for test set
        test_config = self.config.copy()
        test_config["subsample_other"] = False
        test_config["other_subsample_ratio"] = 1.0
        
        test_dataset = ActionDataset(
            folder_path=self.test_folder, 
            config=test_config,
            file_list=self.test_file_list
        )
        self.test_dataset = test_dataset

        testloader_params = {
            "batch_size": self.batch_size, 
            "shuffle": False, 
            "num_workers": self.num_workers, 
            "drop_last": self.drop_last
        }

        return DataLoader(test_dataset, **testloader_params)

    def on_training_epoch_end(self, outputs):
        """
        Called at the end of the training epoch.

        Args:
            outputs (list): List of outputs from the training steps.

        Returns:
            None
        """
        avg_loss = torch.stack(self.trainer.callback_metrics["train_loss"]).mean()
        self.log("avg_train_loss", avg_loss, prog_bar=True)

    def display_confusion_mtx(self, mode="test", pretrained=False, exclude_other=False):
        """
        Displays the confusion matrix for the specified mode with accuracy metrics.

        Args:
            mode (str): The mode for which to display the confusion matrix ("val" or "test").
            pretrained (bool): Whether the model is pretrained.
            exclude_other (bool): Whether to exclude the "Other" category from the confusion matrix.

        Returns:
            None
        """
        if mode == "val":
            gts = self.val_gts
            preds = self.val_preds
        elif mode == "test":
            gts = self.test_gts
            preds = self.test_preds

        if len(gts) == 0 or len(preds) == 0:
            print(f"No data available to display {mode} confusion matrix.")
            return

        # all class names and their indices
        all_class_names = sorted(self.label_dict.keys(), key=lambda k: self.label_dict[k])
        
        # filter out "Other" category if requested
        if exclude_other:
            other_class_idx = self.label_dict.get("Other", -1)
            if other_class_idx != -1:
                # filter predictions and ground truth to exclude "Other" class
                mask = np.array(gts) != other_class_idx
                if mask.sum() == 0:
                    print(f"No non-Other samples available for {mode} confusion matrix.")
                    return
                
                filtered_gts = np.array(gts)[mask]
                filtered_preds = np.array(preds)[mask]
                class_names = [name for name in all_class_names if name != "Other"]
                
                # mapping from original indices to filtered indices
                old_to_new_idx = {}
                new_idx = 0
                for old_idx, name in enumerate(all_class_names):
                    if name != "Other":
                        old_to_new_idx[old_idx] = new_idx
                        new_idx += 1
                
                # remap predictions and ground truth to new indices
                remapped_gts = []
                remapped_preds = []
                
                for gt, pred in zip(filtered_gts, filtered_preds):
                    if gt in old_to_new_idx and pred in old_to_new_idx:
                        remapped_gts.append(old_to_new_idx[gt])
                        remapped_preds.append(old_to_new_idx[pred])
                    elif gt in old_to_new_idx:
                        # prediction is for "Other" class, but GT is not "Other"
                        # Skip this sample or handle appropriately
                        print(f"Warning: Prediction {pred} not in filtered classes, GT: {gt}")
                        continue
                
                cm = confusion_matrix(remapped_gts, remapped_preds, labels=range(len(class_names)))
                overall_accuracy = accuracy_score(remapped_gts, remapped_preds)
                
                suffix = "_no_other"
            else:
                # no "Other" class found, proceed normally
                class_names = all_class_names
                cm = confusion_matrix(gts, preds, labels=range(len(class_names)))
                overall_accuracy = accuracy_score(gts, preds)
                suffix = ""
        else:
            # include all classes
            class_names = all_class_names
            cm = confusion_matrix(gts, preds, labels=range(len(class_names)))
            overall_accuracy = accuracy_score(gts, preds)
            suffix = ""
        
        # per-class accuracy
        class_accuracies = []
        for i in range(len(class_names)):
            if cm[i].sum() > 0: 
                class_acc = cm[i, i] / cm[i].sum()
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False, xticks_rotation='vertical')
        im = disp.im_ # for the colorbar, we get the image

        ax_pos = ax.get_position()
        cbar_ax_left = ax_pos.x0
        cbar_ax_bottom = 0.05  #  move the colorbar up/down
        cbar_ax_width = ax_pos.width
        cbar_ax_height = 0.02  # colorbar thickness

        cbar_ax = fig.add_axes([cbar_ax_left, cbar_ax_bottom, cbar_ax_width, cbar_ax_height])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        title_suffix = " (excluding Other)" if exclude_other and suffix == "_no_other" else ""
        ax.set_title(f'{mode.capitalize()} Confusion Matrix{title_suffix}\nOverall Accuracy: {overall_accuracy:.3f}', 
                    fontsize=14, fontweight='bold', pad=20)

        accuracy_text = "Per-Class Accuracy:\n"
        for class_name, acc in zip(class_names, class_accuracies):
            accuracy_text += f"{class_name}: {acc:.3f}\n"
        
        # text box to the right of the confusion matrix
        ax.text(1.02, 1, accuracy_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                    facecolor="lightgray", alpha=0.8))

        plt.xticks(rotation=0)
        plt.tight_layout(rect=[0, 0.08, 1, 1]) # reserve space from bottom=0.08 upwards for the main plot and text  

        os.makedirs(self.model_folder, exist_ok=True)
        filename = f"{mode}_confusion_matrix{suffix}.png"
        filepath = f"{self.model_folder}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        mlflow.log_artifact(filepath)

        print(f"Saved {mode} confusion matrix to {filepath}\n")
        accuracy_description = f"{mode.capitalize()} Overall Accuracy"
        if exclude_other and suffix == "_no_other":
            accuracy_description += " (excluding Other)"
        print(f"{accuracy_description}: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        print("Per-class accuracies:")
        for class_name, acc in zip(class_names, class_accuracies):
            print(f"  {class_name}: {acc:.3f} ({acc*100:.1f}%)")

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch to log metrics and save the model.

        Returns:
            None
        """
        self.save_model()

    def save_model(self):
        """
        Saves the model state to a file.

        Returns:
            None
        """
        torch.save(
            self.model.state_dict(),
            f"{self.model_folder}/action_head_classification_{self.curr_time_str}.pth",
        )
        actual_current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(
            self.model.state_dict(),
            f"{self.model_folder}/action_head_classification_{actual_current_time}.pth",
        )
    def load_model(self, file_path):
        """
        Loads the model state from a file.

        Args:
            file_path (str): Path to the file containing the model state.

        Returns:
            None
        """
        state_dict = torch.load(file_path)
        
        #  loss-related keys and other non-model parameters - not needed for inference
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith(('loss.', 'criterion.')):
                filtered_state_dict[key] = value
        
        self.model.load_state_dict(filtered_state_dict, strict=False)

    def debug_dataset_info(self, dataloader, dataset_name):
        """Debug function to print dataset information."""
        print(f"\n=== {dataset_name} Dataset Info ===")
        print(f"Label dict: {self.label_dict}")
        print(f"Number of classes: {len(self.label_dict)}")
        
        # sample a few batches to see label distribution
        label_counts = {}
        total_samples = 0
        batch_count = 0
        
        for batch in dataloader:
            if batch_count >= 3:  # only sample first 3 batches
                break
            x, y = batch
            for label in y.cpu().numpy():
                label_counts[label] = label_counts.get(label, 0) + 1
                total_samples += 1
            batch_count += 1
        
        print(f"Sample from first 3 batches ({total_samples} samples):")
        for label_idx, count in sorted(label_counts.items()):
            label_name = [k for k, v in self.label_dict.items() if v == label_idx]
            label_name = label_name[0] if label_name else f"Unknown({label_idx})"
            print(f"  Class {label_idx} ({label_name}): {count} samples")
        print("=" * 40)

    def save_results_to_csv(self, metrics_dict, pretrained_model_path):
        """
        Save model evaluation results to CSV file.
        
        Args:
            metrics_dict (dict): Dictionary containing all evaluation metrics
            pretrained_model_path (str): Path to the pretrained model file
        """
        csv_path = self.config.get("results_csv_path", "/home/valira/Projects/maja-mag/data/models/models_movement_classification/ActionHeadClassification/results_bootstrap.csv")
        
        # create results entry
        result_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': pretrained_model_path,
            'model_name': self.config.get("model_name", ""),
            'keypoint_detector': self.config.get("keypoint_detector", ""),
            'architecture': self.config.get("architecture", ""),
            'training_date': self.config.get("training_date", ""),
            'experiment_name': self.config.get("experiment_name", ""),
            **metrics_dict
        }
        
        df_new = pd.DataFrame([result_entry])
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

    def _store_bootstrap_metrics(self, bootstrap_results: Dict, metric_type: str):
        """
        Store bootstrap results for later access/logging
        
        Args:
            bootstrap_results: Dict of metric name to BootstrapResults
            metric_type: Type of metrics ("all_classes", "bd_only", or "main_labels")
        """
        self.bootstrap_results[metric_type] = {}
        
        if metric_type == "main_labels":
            prefix = "test_main_"
        else:
            prefix = "test_"
        
        for metric_name, result in bootstrap_results.items():
            self.bootstrap_results[metric_type][metric_name] = {
                'mean': result.mean,
                'stderr': result.stderr,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper
            }
            
            self.log(f"{prefix}{metric_name}_mean", result.mean)
            self.log(f"{prefix}{metric_name}_stderr", result.stderr)
            self.log(f"{prefix}{metric_name}_ci_lower", result.ci_lower)
            self.log(f"{prefix}{metric_name}_ci_upper", result.ci_upper)

    def _compute_main_label_metrics(self, gts, preds):
        """
        Compute precision, recall, and F1 metrics at the main label level for hierarchical models.
        
        Args:
            gts (np.array): Ground truth sublabel indices
            preds (np.array): Predicted sublabel indices
            
        Returns:
            dict: Dictionary containing main label metrics
        """
        
        main_gts = []
        main_preds = []
        
        for gt, pred in zip(gts, preds):
            if gt in self.sublabel_to_main and pred in self.sublabel_to_main:
                main_gt = self.main_to_idx[self.sublabel_to_main[gt]]
                main_pred = self.main_to_idx[self.sublabel_to_main[pred]]
                main_gts.append(main_gt)
                main_preds.append(main_pred)
        
        if not main_gts:
            return {}
        
        main_gts = np.array(main_gts)
        main_preds = np.array(main_preds)
        main_labels = list(range(len(self.main_labels)))
        
        main_acc = accuracy_score(main_gts, main_preds)
        main_precision = precision_score(main_gts, main_preds, average='macro', labels=main_labels, zero_division=0.0)
        main_recall = recall_score(main_gts, main_preds, average='macro', labels=main_labels, zero_division=0.0)
        main_f1 = f1_score(main_gts, main_preds, average='macro', labels=main_labels, zero_division=0.0)
        
        return {
            'main_acc': main_acc,
            'main_precision': main_precision,
            'main_recall': main_recall,
            'main_f1': main_f1,
            'main_gts': main_gts,
            'main_preds': main_preds
        }


def train_with_lightning(config, extract_keypoints_config=None):
    """
    Trains the model using PyTorch Lightning.

    Args:
        config (dict): Configuration dictionary containing all training parameters.
        extract_keypoints_config (dict, optional): Configuration from keypoints extraction step.
    """
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)
        print(f"Set random seed to: {seed}")

    num_classes = len(config.get("label_dict"))
    
    model = ActionHeadClassification(num_classes=num_classes) # TODO: add avg_history_dim=True and remove permute and mean in inference
    
    lit_model = LitActionHeadClassification(
        model=model, 
        config=config,
        extract_keypoints_config=extract_keypoints_config
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        min_delta=1e-5, 
        patience=config.get("patience", 10)
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.get("model_folder"),
        filename="best_model_{epoch:02d}_{val_loss:.2f}",
        save_top_k=1,  # only the best model
        mode="min",
        save_weights_only=True,  # only the weights (not the entire model state)
    )

    trainer = L.Trainer(
        max_epochs=config.get("epochs", 10), 
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    if config.get("pretrained", False):
        lit_model.load_model(file_path=config.get("pretrained_model_path"))
        print(f"Loaded pretrained model from: {config.get('pretrained_model_path')}")
    else:
        print("Fitting the model...")
        trainer.fit(lit_model)

        # best model pth file
        ckpt = torch.load(checkpoint_callback.best_model_path)
        model_state_dict = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            # filter out model parameters and exclude loss/criterion parameters
            if k.startswith("model."):
                new_key = k.replace("model.", "")
                new_state_dict[new_key] = v
        
        best_model_pth_path = os.path.join(config.get("model_folder"), "best_model.pth")
        torch.save(new_state_dict, best_model_pth_path)
        print("Finished training. Best model saved to: " + best_model_pth_path)

        print(f"Loading best model from: {checkpoint_callback.best_model_path}")
        best_lit_model = LitActionHeadClassification.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=model,
            config=config,
            extract_keypoints_config=extract_keypoints_config
        )

    test_model = lit_model if config.get("pretrained", False) else best_lit_model
    
    # NOTE: COMMENTED OUT SO WE EVALUATE OD VALIDATION SET EVEN WHEN TRAINING
    #if config.get("pretrained", False):
    print("Evaluating on validation set...")
    test_model.val_gts = []
    test_model.val_preds = []
    
    val_dataloader = test_model.val_dataloader()
    trainer.test(test_model, dataloaders=val_dataloader)
    
    val_logged_metrics = trainer.logged_metrics.copy()
    val_metrics_dict = {
        # bootstrap metrics
        'val_precision_mean': val_logged_metrics.get('test_precision_mean', 0.0),
        'val_precision_stderr': val_logged_metrics.get('test_precision_stderr', 0.0),
        'val_precision_ci_lower': val_logged_metrics.get('test_precision_ci_lower', 0.0),
        'val_precision_ci_upper': val_logged_metrics.get('test_precision_ci_upper', 0.0),
        'val_recall_mean': val_logged_metrics.get('test_recall_mean', 0.0),
        'val_recall_stderr': val_logged_metrics.get('test_recall_stderr', 0.0),
        'val_recall_ci_lower': val_logged_metrics.get('test_recall_ci_lower', 0.0),
        'val_recall_ci_upper': val_logged_metrics.get('test_recall_ci_upper', 0.0),
        'val_f1_mean': val_logged_metrics.get('test_f1_mean', 0.0),
        'val_f1_stderr': val_logged_metrics.get('test_f1_stderr', 0.0),
        'val_f1_ci_lower': val_logged_metrics.get('test_f1_ci_lower', 0.0),
        'val_f1_ci_upper': val_logged_metrics.get('test_f1_ci_upper', 0.0),

        # main label bootstrap metrics for validation
        'val_main_precision_mean': val_logged_metrics.get('test_main_precision_mean', 0.0),
        'val_main_precision_stderr': val_logged_metrics.get('test_main_precision_stderr', 0.0),
        'val_main_precision_ci_lower': val_logged_metrics.get('test_main_precision_ci_lower', 0.0),
        'val_main_precision_ci_upper': val_logged_metrics.get('test_main_precision_ci_upper', 0.0),
        'val_main_recall_mean': val_logged_metrics.get('test_main_recall_mean', 0.0),
        'val_main_recall_stderr': val_logged_metrics.get('test_main_recall_stderr', 0.0),
        'val_main_recall_ci_lower': val_logged_metrics.get('test_main_recall_ci_lower', 0.0),
        'val_main_recall_ci_upper': val_logged_metrics.get('test_main_recall_ci_upper', 0.0),
        'val_main_f1_mean': val_logged_metrics.get('test_main_f1_mean', 0.0),
        'val_main_f1_stderr': val_logged_metrics.get('test_main_f1_stderr', 0.0),
        'val_main_f1_ci_lower': val_logged_metrics.get('test_main_f1_ci_lower', 0.0),
        'val_main_f1_ci_upper': val_logged_metrics.get('test_main_f1_ci_upper', 0.0),
    }
    
    # store validation predictions for confusion matrix
    val_gts_copy = test_model.val_gts.copy()
    val_preds_copy = test_model.val_preds.copy()
    
    print("Evaluating on test set...")
    test_model.test_gts = []
    test_model.test_preds = []
    
    test_dataloader = test_model.test_dataloader()
    trainer.test(test_model)
    
    test_logged_metrics = trainer.logged_metrics
    test_metrics_dict = {

        # bootstrap metrics
        'test_precision_mean': test_logged_metrics.get('test_precision_mean', 0.0),
        'test_precision_stderr': test_logged_metrics.get('test_precision_stderr', 0.0),
        'test_precision_ci_lower': test_logged_metrics.get('test_precision_ci_lower', 0.0),
        'test_precision_ci_upper': test_logged_metrics.get('test_precision_ci_upper', 0.0),
        'test_recall_mean': test_logged_metrics.get('test_recall_mean', 0.0),
        'test_recall_stderr': test_logged_metrics.get('test_recall_stderr', 0.0),
        'test_recall_ci_lower': test_logged_metrics.get('test_recall_ci_lower', 0.0),
        'test_recall_ci_upper': test_logged_metrics.get('test_recall_ci_upper', 0.0),
        'test_f1_mean': test_logged_metrics.get('test_f1_mean', 0.0),
        'test_f1_stderr': test_logged_metrics.get('test_f1_stderr', 0.0),
        'test_f1_ci_lower': test_logged_metrics.get('test_f1_ci_lower', 0.0),
        'test_f1_ci_upper': test_logged_metrics.get('test_f1_ci_upper', 0.0),

        # main label bootstrap metrics
        'test_main_precision_mean': test_logged_metrics.get('test_main_precision_mean', 0.0),
        'test_main_precision_stderr': test_logged_metrics.get('test_main_precision_stderr', 0.0),
        'test_main_precision_ci_lower': test_logged_metrics.get('test_main_precision_ci_lower', 0.0),
        'test_main_precision_ci_upper': test_logged_metrics.get('test_main_precision_ci_upper', 0.0),
        'test_main_recall_mean': test_logged_metrics.get('test_main_recall_mean', 0.0),
        'test_main_recall_stderr': test_logged_metrics.get('test_main_recall_stderr', 0.0),
        'test_main_recall_ci_lower': test_logged_metrics.get('test_main_recall_ci_lower', 0.0),
        'test_main_recall_ci_upper': test_logged_metrics.get('test_main_recall_ci_upper', 0.0),
        'test_main_f1_mean': test_logged_metrics.get('test_main_f1_mean', 0.0),
        'test_main_f1_stderr': test_logged_metrics.get('test_main_f1_stderr', 0.0),
        'test_main_f1_ci_lower': test_logged_metrics.get('test_main_f1_ci_lower', 0.0),
        'test_main_f1_ci_upper': test_logged_metrics.get('test_main_f1_ci_upper', 0.0),

    }
    
    # combine validation and test metrics
    all_metrics_dict = {**val_metrics_dict, **test_metrics_dict}

    for key, value in all_metrics_dict.items():
        if isinstance(value, torch.Tensor):
            all_metrics_dict[key] = value.item()
    
    model_path = config.get("pretrained_model_path") if config.get("pretrained", False) else checkpoint_callback.best_model_path
    test_model.save_results_to_csv(all_metrics_dict, model_path)
    test_model.val_gts = val_gts_copy
    test_model.val_preds = val_preds_copy        
    #else:
    #    trainer.test(test_model)

    test_model.display_confusion_mtx(mode="val", exclude_other=False)
    test_model.display_confusion_mtx(mode="val", exclude_other=True)
    test_model.display_confusion_mtx(mode="test", exclude_other=False)
    test_model.display_confusion_mtx(mode="test", exclude_other=True)


def setup_consolidated_labels(config, embeddings_folder):
    """
    Setup consolidated labels based on data distribution and manual rules.
    
    Args:
        config: Training configuration dict
        embeddings_folder: Path to embeddings folder
        
    Returns:
        Tuple of (updated_config, consolidation_report)
    """
    if not config.get("use_sublabels", False):
        return config, "No sublabel consolidation - use_sublabels is False"
    
    if not config.get("use_consolidation", False):
        return config, "No sublabel consolidation - use_consolidation is False"
    
    consolidation_rules = config.get("sublabel_consolidation_rules", {})
    min_samples_threshold = config.get("min_samples_threshold", 10)
    
    original_joined_dict = config.get("joined_label_dict", {})
    
    consolidated_dict, sublabel_dict, report, consolidation_map = consolidate_labels(
        data_folder=embeddings_folder,
        original_joined_dict=original_joined_dict,
        consolidation_rules=consolidation_rules,
        min_samples_threshold=min_samples_threshold
    )
    
    #config["joined_label_dict"] = consolidated_dict
    config["label_dict"] = sublabel_dict
    config["consolidation_map"] = consolidation_map
    
    return config, report

def run(args):
    """
    Main function to run the movement classification pipeline.

    Args:
        args: Command line arguments
    """
    # ------------------------ PRETRAINED MODEL HANDLING ------------------------


    if args.pretrained.lower() == "true":
        if not args.pretrained_model_path:
            raise ValueError("--pretrained_model_path must be provided when --pretrained is true")
        
        # config from pretrained model's folder
        pretrained_model_dir = os.path.dirname(args.pretrained_model_path)
        previous_config_path = os.path.join(pretrained_model_dir, "config.yaml")

        if os.path.exists(previous_config_path):
            with open(previous_config_path, 'r') as file:
                previous_config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Config file not found in embeddings directory: {previous_config_path}")

        config = previous_config.get("action_head_classification_config")

    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f).get("action_head_classification_config")

        embeddings_dir = args.embeddings_dir if args.embeddings_dir else config.get("input_folder")
        print(f"Using embeddings directory: {embeddings_dir}")
        
        previous_config_path = os.path.join(embeddings_dir, "config.yaml")
        # ensure pretrained is not set in config.yaml (should only be via command line)
        if config.get("pretrained", False):
            raise ValueError("'pretrained' should not be set in config.yaml. Use --pretrained command line argument instead.")

    # ------------------------ LOAD PREVIOUS CONFIG ------------------------
    if os.path.exists(previous_config_path):
        with open(previous_config_path, 'r') as file:
            previous_config = yaml.safe_load(file)
            extract_keypoints_config = previous_config.get("extract_keypoints_config", {})
            embeddings_config = previous_config.get("embeddings_config", {})
            print(f"Loaded extract_keypoints_config and embeddings_config from embeddings directory")
    else:
        raise FileNotFoundError(f"Config file not found in embeddings directory: {previous_config_path}")
    
    # ------------------------ EVALUATION MODE HANDLING ------------------------
    if args.pretrained.lower() == "true":
        # pretrained flags
        config["pretrained"] = True
        config["pretrained_model_path"] = args.pretrained_model_path
        print(f"Using pretrained model: {args.pretrained_model_path}")
        print(f"Using embeddings directory: {config.get('input_folder')}")
        
        if args.experiment_name:
            config["experiment_name"] = args.experiment_name
       
        model_name = os.path.basename(pretrained_model_dir)
        config["experiment_name"] = f"eval_{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # ------------------------ OVERRIDE CONFIG WITH COMMAND LINE ARGUMENTS ------------------------
    if args.embeddings_dir is not None:
        config["input_folder"] = args.embeddings_dir
        print(f"Overriding embeddings directory for evaluation: {args.embeddings_dir}")

    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    
    if args.model_name is not None:
        config["model_name"] = args.model_name
    
    if args.keypoint_detector is not None:
        config["keypoint_detector"] = args.keypoint_detector
        
    if args.architecture is not None:
        config["architecture"] = args.architecture
        
    if args.training_date is not None:
        config["training_date"] = args.training_date
    
    if args.results_csv_path is not None:
        config["results_csv_path"] = args.results_csv_path

    if args.loss_function is not None:
        config["loss_function"] = args.loss_function
    
    if args.focal_alpha is not None:
        config["focal_alpha"] = args.focal_alpha
        
    if args.focal_gamma is not None:
        config["focal_gamma"] = args.focal_gamma
    
    if args.hierarchical_main_weight is not None:
        config["hierarchical_main_weight"] = args.hierarchical_main_weight
        
    if args.hierarchical_sub_weight is not None:
        config["hierarchical_sub_weight"] = args.hierarchical_sub_weight

    # ------------------------ SET FOLDERS ------------------------
    if not config.get("train_folder") or not config.get("val_folder") or not config.get("test_folder"):
        if not config.get("train_test_split"):
            config["train_folder"] = os.path.join(config.get("input_folder"), "all")
            config["val_folder"] = os.path.join(config.get("input_folder"), "all")
            config["test_folder"] = os.path.join(config.get("input_folder"), "all")
        else:
            config["train_folder"] = os.path.join(config.get("input_folder"), "train")
            config["val_folder"] = os.path.join(config.get("input_folder"), "val")
            config["test_folder"] = os.path.join(config.get("input_folder"), "test")

    if not config.get("experiment_name"):
        config["experiment_name"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    output_folder = os.path.join(config.get("model_folder"), config.get("experiment_name"))
    os.makedirs(output_folder, exist_ok=True)
    config["model_folder"] = output_folder

    # ------------------------ MLFLOW ------------------------
    mlflow.pytorch.autolog()
    mlflow.set_experiment(experiment_name="movement_classification")

    with mlflow.start_run(run_name=config.get("experiment_name")) as run:
        # in case of pretrained model, we don't need to create sublabels and consolidation map again
        if not config.get("pretrained", False) and config.get("use_sublabels", False):
            if config.get("use_consolidation", False):
                config, consolidation_report = setup_consolidated_labels(config, config.get("train_folder"))
                print(consolidation_report)
            else:
                sublabel_dict = create_sublabel_dict(config.get("joined_label_dict"))
                config["label_dict"] = sublabel_dict
        
            print(f"Created sublabel dict with {len(config['label_dict'])} sublabels: {list(config['label_dict'].keys())}")
        
        train_with_lightning(config, extract_keypoints_config)

        # save the config in the model folder to keep track of the parameters used
        # only if training, not if pretrained
        if not config.get("pretrained", False): 
            with open(os.path.join(config.get("model_folder"), "config.yaml"), "w") as f:
                yaml.dump(config, f)

            new_config = {
                    "extract_keypoints_config": extract_keypoints_config,
                    "embeddings_config": embeddings_config,
                    "action_head_classification_config": config,
                }

            output_path = os.path.join(output_folder, "config.yaml")
            with open(output_path, 'w') as file:
                yaml.dump(new_config, file)

            mlflow.log_artifact(output_path)

if __name__ == "__main__":
    print("Training started...")
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/rg_ai/keypoints_pipeline/config.yaml"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="data/extracted/embeddings_3d/emb",
        help="Directory containing the embeddings data. If provided, overrides the input_folder in config.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="false",
        help="Whether to use pretrained model (true/false)."
    )
    parser.add_argument(
        "--pretrained_model_path", 
        type=str,
        help="Path to pretrained model file."
    )
    parser.add_argument(
        "--experiment_name",
        type=str, 
        help="Name for the experiment."
    )

    parser.add_argument(
        "--results_csv_path",
        type=str,
        help="Path to results csv file."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name/identifier for the model being tested (e.g., RTMO, BlazePose, YOLO11x)."
    )
    
    parser.add_argument(
        "--keypoint_detector",
        type=str,
        #default="RTMO",
        help="Keypoint detector used for this model (e.g., RTMO, BlazePose, YOLO11x)."
    )
    
    parser.add_argument(
        "--architecture",
        type=str,
        help="Model architecture name (e.g., ActionHeadClassification)."
    )
    
    parser.add_argument(
        "--training_date",
        type=str,
        help="Date when the model was trained (e.g., 2025-08-15)."
    )

    parser.add_argument(
        "--loss_function",
        type=str,
        help="Loss function to use (cross_entropy, weighted_cross_entropy, focal, hierarchical)."
    )
    
    parser.add_argument(
        "--focal_alpha",
        type=float,
        #default=1.0,
        help="Alpha parameter for Focal Loss."
    )
    
    parser.add_argument(
        "--focal_gamma", 
        type=float,
        #default=2.0,
        help="Gamma parameter for Focal Loss."
    )

    parser.add_argument(
        "--hierarchical_main_weight",
        type=float,
        #default=0.33,
        help="Weight for main category loss in hierarchical loss."
    )
    
    parser.add_argument(
        "--hierarchical_sub_weight", 
        type=float,
        #default=0.66,
        help="Weight for sublabel loss in hierarchical loss."
    )

    args = parser.parse_args()

    run(args)

