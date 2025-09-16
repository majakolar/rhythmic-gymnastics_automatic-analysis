# Rhythmic Gymnastics AI Analysis Suite

A comprehensive toolkit for automatic analysis and annotation of rhythmic gymnastics videos.

## Modules

### Annotation Tools (`src/rg_ai/annotation_tools/`)
Interactive GUI application for manual video annotation with frame-by-frame playback, movement type selection (Jump, Balance, Rotation), score assignment, and JSON export.

### EDA (`src/rg_ai/EDA/`)
Statistical analysis and visualization of annotation data including movement distributions, temporal patterns, and grade analysis with interactive Plotly visualizations.

### Keypoints Pipeline (`src/rg_ai/keypoints_pipeline/`)
Complete automated analysis pipeline: 2D and 3D keypoint extraction, movement classification, video segmentation, and LSTM-based automatic grading from raw video to scored segments.

## Quick Start

```bash
# Setup environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .

# Run annotation tool
python src/rg_ai/annotation_tools/annotate_db.py

# Run EDA analysis
python src/rg_ai/EDA/eda.py --annotations_folder data/raw/annotations

# Run pipeline for automatic detection of RG body difficulties
bash src/rg_ai/keypoints_pipeline/run_keypoints_pipeline.sh
```

## License

MIT License