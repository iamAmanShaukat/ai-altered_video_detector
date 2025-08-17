import cv2
import numpy as np
from pathlib import Path
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_preprocessor import VideoPreprocessor
import yaml
import argparse
import time

def extract_features(config_path, video_dir, output_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")
    
    frame_count = config["data"]["frame_count"]
    logger.info(f"Using frame_count: {frame_count}")
    
    # Initialize preprocessor with pretrained backbone
    preprocessor = VideoPreprocessor(config, augment=False)
    logger.info("Using pretrained Xception backbone")
    
    # Setup directories
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_start_time = time.time()
    video_count = 0
    
    # Process all videos in directory
    for video_path in video_dir.rglob("*.mp4"):
        video_start_time = time.time()
        try:
            # Determine video type from directory structure
            type_prefix = video_path.parts[-2].lower()
            if type_prefix == "original_sequences":
                type_prefix = "real"

            # Extract frames from video
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while len(frames) < frame_count:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.warning(f"Invalid frame in {video_path}, stopping at {len(frames)} frames")
                    break
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                logger.error(f"No valid frames in {video_path}, skipping")
                continue

            # Pad with last frame if we don't have enough frames
            if len(frames) < frame_count:
                logger.warning(f"Video {video_path} has {len(frames)} frames, padding with last frame")
                frames.extend([frames[-1]] * (frame_count - len(frames)))
            frames = np.array(frames[:frame_count], dtype=np.uint8)

            # Extract features using preprocessor
            frames_np, appearance_features, flow_maps = preprocessor.extract_features(frames)

            # Create output filename based on video type
            if type_prefix == "real":
                output_name = f"real_{video_path.stem}.npz"
            else:
                parts = video_path.stem.split('_')
                target_id = parts[0]  # Use only the first part for manipulated videos
                output_name = f"{type_prefix}_{target_id}.npz"

            # Save features to .npz file
            output_path = output_dir / output_name
            if output_path.exists():
                logger.warning(f"Output file {output_path} already exists, overwriting")
            
            np.savez(
                output_path,
                frames=frames_np,
                appearance=appearance_features,
                flow=flow_maps
            )
            
            video_count += 1
            logger.info(f"Processed {video_path} -> {output_path} in {time.time() - video_start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")

    total_time = time.time() - total_start_time
    logger.info(f"Processed {video_count} videos in {total_time:.3f}s")
    return video_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from videos using Xception backbone")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--video_dir", required=True, help="Directory with videos")
    parser.add_argument("--output_dir", required=True, help="Directory for .npz files")
    args = parser.parse_args()
    
    extract_features(args.config, args.video_dir, args.output_dir)