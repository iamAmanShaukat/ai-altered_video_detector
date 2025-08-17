#!/usr/bin/env python3
"""
Comprehensive visualization tool for deepfake detection features
Combines feature space visualization, label checking, and frame debugging
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
import os
import cv2
import argparse

def setup_logging():
    """Set up logging with timestamps and appropriate formatting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """Load the project configuration from the standard location"""
    # Navigate to project root to find the config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_features(feature_dir, class_map):
    """Load and validate pre-extracted features for visualization"""
    logger = logging.getLogger(__name__)
    
    # Find all feature files in the specified directory
    feature_files = list(feature_dir.glob('*.npz'))
    if not feature_files:
        logger.error("No feature files found.")
        return None, None, None, None
    
    # Initialize containers for different feature types
    all_app_features = []    # Appearance features from Xception
    all_flow_features = []   # Optical flow features (if available)
    all_labels = []          # Class labels as integers
    file_labels = []         # Original filenames for reference
    
    # Process each feature file
    for npz_file in feature_files:
        try:
            # Load the NPZ file containing extracted features
            data = np.load(npz_file)
            
            # Skip files missing appearance features
            if 'appearance' not in data:
                logger.warning(f"Missing appearance in {npz_file.name}, skipping")
                continue
                
            # Extract and average appearance features across frames
            appearance = data['appearance']
            app_feature_vec = np.mean(appearance, axis=0)  # Average over temporal dimension
            
            # Process optical flow features if available
            flow_feature_vec = None
            if 'flow' in data:
                flow = data['flow']
                # Flatten spatial dimensions and average over frames
                flow_feature_vec = np.mean(flow.reshape(flow.shape[0], -1), axis=0)
            
            # Extract class label from filename using naming convention
            fname = npz_file.name
            if fname.startswith('real_'):
                label = class_map['real']
            else:
                # For deepfakes, the prefix indicates the manipulation type
                prefix = fname.split('_')[0]
                if prefix in class_map:
                    label = class_map[prefix]
                else:
                    label = -1  # Unknown class
                    
            # Skip files with unknown classes
            if label == -1:
                logger.warning(f"Unknown class for {npz_file.name}, skipping")
                continue
                
            # Store the processed features and labels
            all_app_features.append(app_feature_vec)
            if flow_feature_vec is not None:
                all_flow_features.append(flow_feature_vec)
            all_labels.append(label)
            file_labels.append(fname)
            
        except Exception as e:
            logger.error(f"Could not process {npz_file.name}: {e}")
    
    # Verify we have at least some valid features
    if not all_app_features:
        logger.error("No valid features found.")
        return None, None, None, None
    
    # Convert lists to numpy arrays for further processing
    all_app_features = np.array(all_app_features)
    all_flow_features = np.array(all_flow_features) if all_flow_features else None
    all_labels = np.array(all_labels)
    
    logger.info(f"Loaded {len(all_app_features)} feature vectors.")
    return all_app_features, all_flow_features, all_labels, file_labels

def visualize_feature_space(app_features, flow_features, labels, class_map, output_dir):
    """Create visualizations of feature spaces using dimensionality reduction techniques
    
    Generates both t-SNE and PCA plots for appearance features and optical flow features (if available)
    to help understand the distribution and separability of different classes in the feature space.
    """
    logger = logging.getLogger(__name__)
    
    # Need at least two samples to perform dimensionality reduction
    if len(app_features) < 2:
        logger.error("Not enough features to visualize.")
        return
    
    # Organize features into datasets for visualization
    datasets = [("Appearance", app_features)]  # Always include appearance features
    if flow_features is not None:
        datasets.append(("Flow", flow_features))  # Add flow features if available
    
    # Set up the figure layout - one row per feature type, two columns for t-SNE and PCA
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 2, figsize=(14, 6 * num_datasets))
    if num_datasets == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D even with only one dataset
    
    # Define colors for different classes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Process each feature type (appearance and possibly flow)
    for dataset_idx, (dataset_name, features) in enumerate(datasets):
        logger.info(f"Processing {dataset_name} features...")
        
        # Apply t-SNE for non-linear dimensionality reduction
        logger.info(f"Running t-SNE for {dataset_name}...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        features_tsne = tsne.fit_transform(features)
        
        # Apply PCA for linear dimensionality reduction
        logger.info(f"Running PCA for {dataset_name}...")
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        # Create t-SNE scatter plot
        ax_tsne = axes[dataset_idx, 0]
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            # Convert numeric class index back to class name
            class_name = list(class_map.keys())[list(class_map.values()).index(class_idx)]
            ax_tsne.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                           s=20, alpha=0.7, label=class_name, color=colors[class_idx % len(colors)])
        ax_tsne.set_title(f'{dataset_name} Feature Space (t-SNE)')
        ax_tsne.set_xlabel('Component 1')
        ax_tsne.set_ylabel('Component 2')
        ax_tsne.legend()
        
        # Create PCA scatter plot
        ax_pca = axes[dataset_idx, 1]
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            class_name = list(class_map.keys())[list(class_map.values()).index(class_idx)]
            ax_pca.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                          s=20, alpha=0.7, label=class_name, color=colors[class_idx % len(colors)])
        ax_pca.set_title(f'{dataset_name} Feature Space (PCA)')
        ax_pca.set_xlabel('Component 1')
        ax_pca.set_ylabel('Component 2')
        ax_pca.legend()
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the visualization to disk
    output_path = output_dir / 'feature_space_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature space plot saved as: {output_path}")
    plt.show()  # Also display the plot if in interactive mode

def visualize_sample_frames(feature_dir, class_map, output_dir, samples_per_class=3):
    """Display representative frames from each class to verify correct labeling
    
    Extracts and displays a few sample frames from each class to help visually confirm
    that videos are correctly labeled and that the features make sense.
    """
    logger = logging.getLogger(__name__)
    
    # ImageNet normalization constants used during feature extraction
    # We need to reverse this normalization to display images properly
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def unnormalize(img):
        """Convert normalized image back to displayable RGB format"""
        return np.clip((img * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)
    
    # Find representative samples for each class
    sample_files = {label: [] for label in class_map}
    for npz_file in feature_dir.glob('*.npz'):
        fname = npz_file.name
        
        # Determine class from filename using naming convention
        if fname.startswith('real_'):
            label = 'real'
        else:
            prefix = fname.split('_')[0]
            label = prefix if prefix in class_map else None
            
        # Add file to samples if we need more examples of this class
        if label and len(sample_files[label]) < samples_per_class:
            sample_files[label].append(npz_file)
            
        # Stop searching once we have enough samples for all classes
        if all(len(files) == samples_per_class for files in sample_files.values()):
            break
    
    # Create a grid visualization with samples from each class
    plt.figure(figsize=(12, 12))
    for row_idx, label in enumerate(class_map):
        files = sample_files[label]
        for col_idx, npz_file in enumerate(files):
            try:
                # Load the feature file and extract the first frame
                data = np.load(npz_file)
                frames = data['frames']
                frame = frames[0] if frames.shape[0] > 0 else None
                
                if frame is not None:
                    # Log frame statistics for debugging
                    logger.info(f"Class: {label}, File: {npz_file.name}")
                    logger.info(f"  Frame 0 stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.4f}, std={frame.std():.4f}, dtype={frame.dtype}")
                    
                    # Convert normalized float images back to displayable format
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        frame_disp = unnormalize(frame)
                    else:
                        frame_disp = frame
                    
                    # Add the frame to the grid
                    ax = plt.subplot(len(class_map), samples_per_class, row_idx * samples_per_class + col_idx + 1)
                    ax.imshow(frame_disp)
                    ax.set_title(f"{label}\n{npz_file.name}")
                    ax.axis('off')
            except Exception as e:
                logger.error(f"Error processing {npz_file}: {e}")
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the visualization to disk
    output_path = output_dir / 'sample_frames.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sample frames plot saved as: {output_path}")
    plt.show()  # Also display the plot if in interactive mode

def debug_face_detection(video_path, frame_idx=4):
    """Display a specific frame from a video to help debug face detection issues
    
    Loads a video file, extracts all frames, and displays the specified frame
    to help diagnose problems with face detection algorithms.
    """
    logger = logging.getLogger(__name__)
    
    # Verify the video file exists
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Open the video and read all frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    success = True
    
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)
    cap.release()
    
    # Display the requested frame if it exists
    if frame_idx < len(frames):
        frame = frames[frame_idx]
        plt.figure(figsize=(10, 8))
        # Convert from BGR (OpenCV format) to RGB (matplotlib format)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_idx} from {video_path.name}")
        plt.axis('off')
        plt.show()
        logger.info(f"Displayed frame {frame_idx} from video with {len(frames)} total frames")
    else:
        logger.error(f"Frame {frame_idx} not found in {video_path} (video has {len(frames)} frames)")

def main():
    """Main entry point for the feature visualization pipeline
    
    Orchestrates the entire visualization process, including loading features,
    generating feature space visualizations, and displaying sample frames from each class.
    """
    # Set up logging for the visualization process
    logger = setup_logging()
    logger.info("Starting comprehensive feature visualization...")
    
    # Load project configuration from config file
    config = load_config()
    # Construct path to the feature directory from project root
    feature_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), config['data']['feature_dir']))
    # Get class mapping from config (e.g., 'real': 0, 'deepfake': 1)
    class_map = config['data']['class_map']
    
    # Ensure output directory exists for saving visualizations
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_dir.mkdir(exist_ok=True)
    
    # Load pre-extracted features from NPZ files
    app_features, flow_features, labels, file_labels = load_features(feature_dir, class_map)
    if app_features is None:
        logger.error("Failed to load features. Exiting.")
        return
    
    # Generate t-SNE and PCA visualizations of the feature space
    logger.info("Creating feature space visualizations...")
    visualize_feature_space(app_features, flow_features, labels, class_map, output_dir)
    
    # Display sample frames from each class for visual verification
    logger.info("Creating sample frame visualizations...")
    visualize_sample_frames(feature_dir, class_map, output_dir)
    
    # Optional: Show a frame from the first video to help debug face detection
    if len(file_labels) > 0:
        logger.info("Debug face detection for first video...")
        # Assume the corresponding video file has the same name but .mp4 extension
        first_video = feature_dir / file_labels[0].replace('.npz', '.mp4')
        if first_video.exists():
            debug_face_detection(first_video)
    
    logger.info("Visualization completed successfully!")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Comprehensive feature visualization tool")
    parser.add_argument("--mode", choices=["all", "features", "frames", "debug"], default="all",
                       help="Visualization mode: all=complete visualization, features=feature space only, frames=sample frames only, debug=examine specific video")
    parser.add_argument("--video_path", help="Path to specific video for debugging (required in debug mode)")
    parser.add_argument("--frame_idx", type=int, default=4, help="Frame index to display when debugging (default: 4)")
    
    args = parser.parse_args()
    
    # Handle different execution modes
    if args.mode == "debug" and args.video_path:
        # Debug mode: examine a specific frame from a specific video
        setup_logging()
        debug_face_detection(Path(args.video_path), args.frame_idx)
    else:
        # Standard mode: run the complete visualization pipeline
        main()