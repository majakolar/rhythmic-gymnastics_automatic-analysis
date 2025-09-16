import os
from sympy import N
import torch
import torch.nn as nn
import argparse

from rg_ai.keypoints_pipeline.movement_classification import ActionHeadClassification
from rg_ai.keypoints_pipeline.testing.test_video import load_config_from_yaml


class ActionHeadClassificationONNX(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    
    def forward(self, feat):
        return self.model(feat, avg_history_dim=True) # NOTE: avg_history_dim=True is needed for the ONNX model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_folder",
        type=str,
        default="data/models/models_movement_classification/ActionHeadClassification/ahs.pth",
        help="Path to action model folder",
    )

    args = parser.parse_args()

    config = load_config_from_yaml(args.model_folder)

    n_classes = len(config.label_dict)  # NOTE: be careful to change num_classes here
    model = ActionHeadClassification(num_classes=n_classes)
    model.load_state_dict(
        torch.load(
            os.path.join(config.action_model_path)
        )
    )
    
    # ONNX-compatible wrapper
    onnx_model = ActionHeadClassificationONNX(model)
    onnx_model.eval()

    output_folder = os.path.join(args.model_folder, "onnx")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "best_model.onnx")

    N_frames = config.embeddings_config["n_frames_before"] + config.embeddings_config["n_frames_after"] + 1

    dummy_input = torch.randn(1, N_frames, 17, 512) #.half()
    print(onnx_model(dummy_input), onnx_model(dummy_input).shape)

    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    print("Model exported to onnx")
