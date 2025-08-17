import torch
import numpy as np
from pathlib import Path
import logging

class VideoDataset:
    def __init__(self, video_paths, labels, feature_dir, multi_class, class_map, frame_count=8, image_size=224, augment=False):
        # Initialize dataset with video paths and corresponding labels
        self.video_paths = video_paths
        self.labels = labels
        self.feature_dir = Path(feature_dir)
        self.multi_class = multi_class  # Whether to use multi-class classification
        self.class_map = class_map      # Mapping between class names and indices
        self.frame_count = frame_count  # Number of frames to process per video
        self.image_size = image_size    # Size of input images
        self.augment = augment          # Whether to apply data augmentation
        self.logger = logging.getLogger(__name__)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Get video path and label for the requested index
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Determine feature filename based on video type
        type_prefix = video_path.parent.name
        if type_prefix == "real":
            # Real videos use a simple naming convention
            feature_name = f"real_{video_path.stem}.npz"
        else:
            # Deepfake videos extract the target ID from filename
            parts = video_path.stem.split('_')
            target_id = parts[0]
            feature_name = f"{type_prefix}_{target_id}.npz"
            
        # Construct full path to feature file
        feature_path = self.feature_dir / feature_name
        if not feature_path.exists():
            self.logger.error(f"Feature file {feature_path} not found for video {video_path}")
            raise FileNotFoundError(f"Feature file {feature_path} not found")
            
        try:
            # Load pre-extracted features from NPZ file
            features = np.load(feature_path)
            data = {
                "appearance": torch.from_numpy(features["appearance"]).float()
            }
            
            # Add optical flow features if available
            if "flow" in features:
                data["flow"] = torch.from_numpy(features["flow"]).float()
                # Handle flow feature count mismatch
                if data["flow"].shape[0] != self.frame_count - 1:
                    self.logger.warning(f"Flow count mismatch in {feature_path}: {data['flow'].shape[0]} vs {self.frame_count-1}")
                    data["flow"] = data["flow"][:self.frame_count-1]
            
            # Handle appearance feature count mismatch
            if data["appearance"].shape[0] != self.frame_count:
                self.logger.warning(f"Frame count mismatch in {feature_path}: {data['appearance'].shape[0]} vs {self.frame_count}")
                data["appearance"] = data["appearance"][:self.frame_count]
            
            return data, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Error loading features for {video_path}: {e}")
            raise