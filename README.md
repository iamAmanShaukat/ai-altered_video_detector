# Deepfake Video Detector

A deep learning-based system for detecting deepfake videos using Xception backbone with frame-level classification.

## Project Structure

```
deepfake_video_detector/
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── configs/
│   └── config.yaml            # Configuration file
├── data/
│   ├── videos/                # Input video directory
│   └── features/              # Extracted features directory
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── xception_model.py  # Xception model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── video_dataset.py   # Dataset class
│   │   ├── video_preprocessor.py  # Video preprocessing
│   │   ├── extract_features.py    # Feature extraction
│   │   ├── feature_quality_check.py  # Feature validation
│   │   └── feature_validator.py     # Feature validation utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── main.py           # Main training script
│   │   └── trainer.py        # Training utilities
│   └── visualization/
│       ├── __init__.py
│       ├── visualize_features.py  # Feature visualization
│       ├── visualize_manual_label_check.py  # Label verification
│       ├── visualize_no_face_frame.py  # Face detection debug
│       └── *.png             # Generated plots
├── logs/                      # Training logs
├── checkpoints/               # Model checkpoints
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deepfake_video_detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml` to configure:
- Data paths and parameters
- Model architecture settings
- Training hyperparameters
- Augmentation settings
- Face detection parameters

## Usage

### Training

```bash
# Train the model
python main.py --mode train

# Evaluate the model
python main.py --mode evaluate
```

### Feature Extraction

```bash
python src/data/extract_features.py --config configs/config.yaml --video_dir data/videos --output_dir data/features
```

**Note**: The system requires pre-extracted features from the feature extraction pipeline. Run the above command first to generate features before training.



### Visualization

```bash
# Visualize feature space
python src/visualization/visualize_features.py

# Check manual labels
python src/visualization/visualize_manual_label_check.py
```

## Data Organization

Place your videos in the following structure:
```
data/videos/
├── real/
│   └── *.mp4
├── deepfake/
│   └── *.mp4
├── face2face/
│   └── *.mp4
└── neuraltextures/
    └── *.mp4
```

## Model Architecture

- **Backbone**: Xception network (pretrained on ImageNet) for spatial feature extraction
- **Temporal Modeling**: Global average pooling over pre-extracted features
- **Classification**: Multi-class classifier for 4 categories (real, deepfake, face2face, neuraltextures)

**Note**: The system uses the pretrained Xception backbone by default. If you have a fine-tuned backbone (`checkpoints/finetuned_backbone.pth`), it will be used automatically for better performance.



