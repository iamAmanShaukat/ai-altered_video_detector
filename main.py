#!/usr/bin/env python3

import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os
import yaml
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

# Add src directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data.video_dataset import VideoDataset
from models.xception_model import XceptionModel
from utils.trainer import Trainer


def setup_logging():
    """Set up logging to both console and file"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def load_config(config_path=None):
    """Load and validate the YAML configuration file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Make sure all required sections exist
    required = ['data', 'model', 'training']
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return config


def setup_device(config):
    """Set up the computation device (GPU/CPU)"""
    device_name = config["training"]["device"]
    
    # Fall back to CPU if CUDA requested but not available
    if device_name == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU instead")
        device_name = "cpu"
    
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")
    return device


def prepare_data(config):
    """Prepare video data and create feature directory if needed"""
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Get paths from config
    video_dir = base_dir / config["data"]["video_dir"]
    feature_dir = base_dir / config["data"]["feature_dir"]
    
    # Verify video directory exists
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Create feature directory if it doesn't exist
    if not feature_dir.exists():
        feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MP4 videos recursively
    video_paths = list(video_dir.rglob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    
    # Extract labels from directory structure
    labels = []
    valid_paths = []
    
    for video_path in video_paths:
        try:
            # Get category from parent directory name
            category = video_path.parts[-2]
            if category == "original_sequences":
                category = "real"
            
            # Only include videos with known categories
            if category in config["data"]["class_map"]:
                labels.append(config["data"]["class_map"][category])
                valid_paths.append(video_path)
        except Exception as e:
            logging.warning(f"Error processing {video_path}: {e}")
    
    if not valid_paths:
        raise ValueError("No valid videos found")
    
    logging.info(f"Found {len(valid_paths)} videos")
    logging.info(f"Class distribution: {Counter(labels)}")
    
    return valid_paths, labels, feature_dir


def split_data(video_paths, labels, config):
    """Split data into train, validation and test sets with stratification"""
    test_split = config["data"]["test_split"]
    val_split = config["data"]["val_split"]
    
    # First split off the training set
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        video_paths, labels, test_size=test_split, stratify=labels, random_state=42
    )
    
    # Then split the remaining data into validation and test sets
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, test_size=val_split, stratify=val_test_labels, random_state=42
    )
    
    logging.info(f"Split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    
    return (train_paths, val_paths, test_paths, 
            train_labels, val_labels, test_labels)


def create_datasets_and_loaders(train_data, val_data, test_data, feature_dir, config):
    """Create VideoDataset objects and their corresponding DataLoaders"""
    # Set up common parameters for all datasets
    dataset_params = {
        'feature_dir': feature_dir,
        'multi_class': True,
        'class_map': config["data"]["class_map"],
        'frame_count': config["data"]["frame_count"],
        'image_size': config["data"]["image_size"]
    }
    
    # Create datasets (only apply augmentation to training data)
    train_dataset = VideoDataset(video_paths=train_data[0], labels=train_data[1], augment=True, **dataset_params)
    val_dataset = VideoDataset(video_paths=val_data[0], labels=val_data[1], augment=False, **dataset_params)
    test_dataset = VideoDataset(video_paths=test_data[0], labels=test_data[1], augment=False, **dataset_params)
    
    # Create data loaders with appropriate settings
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def main(config_path=None, mode="train"):
    """Main entry point for training or evaluation of the deepfake detection model"""
    try:
        # Initialize logging and environment
        setup_logging()
        logging.info("Starting Deepfake Video Detector Pipeline")
        
        # Load configuration and set up hardware
        config = load_config(config_path)
        device = setup_device(config)
        
        # Data preparation pipeline
        video_paths, labels, feature_dir = prepare_data(config)
        train_data, val_data, test_data = split_data(video_paths, labels, config)
        train_loader, val_loader, test_loader = create_datasets_and_loaders(
            train_data, val_data, test_data, feature_dir, config
        )
        
        # Model initialization
        model = XceptionModel(config).to(device)
        trainer = Trainer(config, model, train_loader=train_loader, val_loader=val_loader)
        
        # Run either training or evaluation mode
        if mode == "train":
            logging.info("Starting training...")
            trainer.train()
            
            # Final evaluation on test set
            logging.info("=" * 60)
            logging.info("FINAL TEST SET EVALUATION")
            logging.info("=" * 60)
            test_accuracy = trainer.evaluate(test_loader)
            logging.info(f"Final Test Accuracy: {test_accuracy:.2f}%")
            
        else:
            logging.info("Starting evaluation...")
            
            # Run evaluation on validation set
            logging.info("=" * 60)
            logging.info("VALIDATION SET EVALUATION")
            logging.info("=" * 60)
            val_accuracy = trainer.evaluate(val_loader)
            
            # Run evaluation on test set
            logging.info("=" * 60)
            logging.info("TEST SET EVALUATION")
            logging.info("=" * 60)
            test_accuracy = trainer.evaluate(test_loader)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Video Detector")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train",
                       help="Mode: train or evaluate")
    parser.add_argument("--config", help="Path to config file (optional)")
    args = parser.parse_args()
    
    main(config_path=args.config, mode=args.mode)
