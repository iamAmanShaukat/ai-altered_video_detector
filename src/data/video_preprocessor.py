import torch
import torchvision.transforms as T
import timm
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import logging
import time

class VideoPreprocessor:
    """Handles video frame processing, face detection, and feature extraction."""
    def __init__(self, config, augment=False):
        """Initialize the video preprocessor with configuration settings.
        
        Args:
            config: Dictionary containing model and processing parameters
            augment: Whether to apply data augmentation to frames
        """
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.frame_count = config["data"]["frame_count"]
        self.image_size = config["data"]["image_size"]
        
        # Initialize backbone model for feature extraction
        self.backbone = timm.create_model(
            config["model"]["architecture"],
            pretrained=True,
            num_classes=0  # Remove classification head
        ).to(self.device)
        self.backbone.eval()
        
        self.augment = augment
        
        # Define augmentation pipeline for training
        self.augment_transform = T.Compose([
            T.ToPILImage(),
            T.ColorJitter(
                brightness=config["augmentation"]["brightness"],
                contrast=config["augmentation"]["contrast"],
                saturation=config["augmentation"]["saturation"],
                hue=config["augmentation"]["hue"]
            ),
            T.RandomResizedCrop(
                self.image_size,
                scale=(config["augmentation"]["crop_scale_min"], config["augmentation"]["crop_scale_max"])
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.GaussianBlur(
                kernel_size=3,
                sigma=(config["augmentation"]["blur_sigma_min"], config["augmentation"]["blur_sigma_max"])
            ),
            T.Normalize(mean=config["normalization"]["imagenet_mean"], std=config["normalization"]["imagenet_std"]),
        ])
        
        # Define standard transformation pipeline for inference
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=config["normalization"]["imagenet_mean"], std=config["normalization"]["imagenet_std"]),
        ])
        
        # Initialize face detector
        self.face_detector = MTCNN(
            image_size=self.image_size,
            margin=config["face_detection"]["margin"],
            min_face_size=config["face_detection"]["min_face_size"],
            thresholds=config["face_detection"]["thresholds"],
            device=self.device
        )
        self.logger = logging.getLogger(__name__)

    def load_backbone_weights(self, weights_path):
        """Load pre-trained or fine-tuned weights for the backbone model.
        
        Args:
            weights_path: Path to the model weights file
        """
        self.backbone.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.logger.info(f"Loaded fine-tuned backbone weights from {weights_path}")

    def detect_face_fallback(self, frame_rgb):
        """Use OpenCV's Haar cascade as a fallback when MTCNN fails to detect faces.
        
        Args:
            frame_rgb: RGB image as numpy array
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates for face bounding box or None
        """
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return x, y, x + w, y + h
        except Exception as e:
            self.logger.warning(f"OpenCV face detection fallback failed: {e}")
        return None

    def extract_features(self, frames):
        """Process video frames to extract appearance and motion features.
        
        Args:
            frames: List or array of video frames
            
        Returns:
            Tuple of (processed_frames, appearance_features, optical_flow_maps)
        """
        start_time = time.time()
        try:
            cropped_frames = []
            flow_maps = []  
            faces_detected = 0
            
            # Process each frame individually
            for i, frame in enumerate(frames):
                frame_start_time = time.time()
                
                # Validate frame integrity
                if frame is None or frame.size == 0:
                    self.logger.error(f"Frame {i} is None or empty")
                    continue
                    
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    self.logger.error(f"Frame {i} invalid shape {frame.shape}, expected (H, W, 3)")
                    continue
                    
                if frame.dtype != np.uint8:
                    self.logger.warning(f"Frame {i} dtype {frame.dtype}, converting to uint8")
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Convert BGR to RGB color space
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    self.logger.error(f"Frame {i} BGR2RGB failed: {str(e)}, trying grayscale")
                    try:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
                    except cv2.error:
                        self.logger.error(f"Frame {i} grayscale conversion failed")
                        continue
                # Ensure memory layout is contiguous for efficient processing
                if not frame_rgb.flags['C_CONTIGUOUS']:
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                    
                # Detect and crop to face region
                try:
                    # Try primary face detection with MTCNN
                    boxes, _ = self.face_detector.detect(frame_rgb)
                    if boxes is not None and len(boxes) > 0:
                        faces_detected += 1
                        x1, y1, x2, y2 = map(int, boxes[0])
                        x1, y1 = max(0, x1), max(0, y1)  # Ensure non-negative coordinates
                        frame_cropped = frame_rgb[y1:y2, x1:x2]
                        
                        # Check if crop is valid
                        if frame_cropped.size == 0 or frame_cropped.shape[0] == 0 or frame_cropped.shape[1] == 0:
                            self.logger.warning(f"Frame {i} cropped to empty image, using full frame")
                            frame_cropped = frame_rgb
                    else:
                        # Try OpenCV fallback face detection
                        fallback_box = self.detect_face_fallback(frame_rgb)
                        if fallback_box is not None:
                            x1, y1, x2, y2 = fallback_box
                            faces_detected += 1
                            frame_cropped = frame_rgb[y1:y2, x1:x2]
                            self.logger.info(f"OpenCV fallback used for frame {i}")
                            
                            # Check if fallback crop is valid
                            if frame_cropped.size == 0 or frame_cropped.shape[0] == 0 or frame_cropped.shape[1] == 0:
                                self.logger.warning(f"Frame {i} OpenCV crop empty, using full frame")
                                frame_cropped = frame_rgb
                        else:
                            self.logger.warning(f"No face detected in frame {i}, using full frame")
                            frame_cropped = frame_rgb
                except Exception as e:
                    self.logger.error(f"Face detection failed for frame {i}: {str(e)}")
                    frame_cropped = frame_rgb
                
                # Resize to standard dimensions
                frame_cropped = cv2.resize(frame_cropped, (self.image_size, self.image_size))
                
                # Apply augmentation if enabled
                if self.augment:
                    frame_cropped = self.augment_transform(frame_cropped).permute(1, 2, 0).numpy()
                    frame_cropped = np.clip(frame_cropped * 255, 0, 255).astype(np.uint8)
                    
                cropped_frames.append(frame_cropped)
                self.logger.debug(f"Frame {i} processed in {time.time() - frame_start_time:.3f}s")
            # Handle case where no valid frames were processed
            if not cropped_frames:
                self.logger.error("No valid frames processed, returning zeros")
                return (np.zeros((self.frame_count, self.image_size, self.image_size, 3), dtype=np.float32),
                        np.zeros((self.frame_count, self.config["model"]["feature_dim"]), dtype=np.float32),
                        np.zeros((self.frame_count-1, self.image_size, self.image_size, 2), dtype=np.float32))
            
            # Pad sequence if we don't have enough frames
            while len(cropped_frames) < self.frame_count:
                self.logger.warning(f"Padding with last frame, only {len(cropped_frames)} valid frames")
                cropped_frames.append(cropped_frames[-1])
                
            # Ensure we have exactly the required number of frames
            cropped_frames = cropped_frames[:self.frame_count]
            frames = np.array(cropped_frames, dtype=np.uint8)
            
            # Calculate optical flow between consecutive frames
            for i in range(1, len(cropped_frames)):
                # Convert to grayscale for optical flow calculation
                prev_frame = cv2.cvtColor(cropped_frames[i-1], cv2.COLOR_RGB2GRAY)
                curr_frame = cv2.cvtColor(cropped_frames[i], cv2.COLOR_RGB2GRAY)
                
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None, 
                    pyr_scale=self.config["optical_flow"]["pyr_scale"],
                    levels=self.config["optical_flow"]["levels"],
                    winsize=self.config["optical_flow"]["winsize"],
                    iterations=self.config["optical_flow"]["iterations"],
                    poly_n=self.config["optical_flow"]["poly_n"],
                    poly_sigma=self.config["optical_flow"]["poly_sigma"],
                    flags=self.config["optical_flow"]["flags"]
                )
                
                # Ensure flow map has consistent dimensions
                flow = cv2.resize(flow, (self.image_size, self.image_size))
                flow_maps.append(flow)
            # Pad optical flow maps if needed
            while len(flow_maps) < self.frame_count - 1:
                self.logger.warning(f"Padding flow with last flow map, only {len(flow_maps)} valid flows")
                flow_maps.append(flow_maps[-1])
            flow_maps = np.array(flow_maps[:self.frame_count-1], dtype=np.float32)
            
            # Convert frames to tensors and apply normalization
            frames_tensor = torch.stack([self.transform(frame).to(self.device) for frame in frames])
            frames_np = frames_tensor.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Extract features using backbone model
            with torch.no_grad():
                features = self.backbone(frames_tensor)
            features_np = features.cpu().numpy()
            
            # Validate output data quality
            if np.all(frames_np == 0):
                raise ValueError("All frames are zero after preprocessing")
            if np.all(features_np == 0):
                raise ValueError("All features are zero after extraction")
            if np.isnan(frames_np).any() or np.isinf(frames_np).any():
                raise ValueError("NaN or Inf detected in frames")
            if np.isnan(features_np).any() or np.isinf(features_np).any():
                raise ValueError("NaN or Inf detected in features")
            if np.isnan(flow_maps).any() or np.isinf(flow_maps).any():
                raise ValueError("NaN or Inf detected in optical flow")
            # Calculate standard deviation for quality checks
            frames_std = np.std(frames_np)
            features_std = np.std(features_np)
            flow_std = np.std(flow_maps)
            
            # Check for high standard deviation (potential quality issues)
            if frames_std > self.config["training"]["frames_std_threshold"]:
                self.logger.warning(f"Frames std too high: {frames_std:.4f}")
            if flow_std > self.config["training"]["frames_std_threshold"]:
                self.logger.warning(f"Flow std too high: {flow_std:.4f}")
                
            # Check for uniform data (indicates processing failure)
            if np.all(frames_np == frames_np.flat[0]):
                raise ValueError("All frames have the same value after preprocessing")
            if np.all(features_np == features_np.flat[0]):
                raise ValueError("All features have the same value after extraction")
            if np.all(flow_maps == flow_maps.flat[0]):
                raise ValueError("All flow maps have the same value")
                
            # Log processing statistics
            total_time = time.time() - start_time
            self.logger.info(f"Processed {len(cropped_frames)} frames, {faces_detected} faces detected, {len(flow_maps)} flow maps in {total_time:.3f}s")
            self.logger.debug(f"Frames shape: {frames_np.shape}, std: {frames_std:.4f}, mean: {np.mean(frames_np):.4f}")
            self.logger.debug(f"Features shape: {features_np.shape}, std: {features_std:.4f}, mean: {np.mean(features_np):.4f}")
            self.logger.debug(f"Flow shape: {flow_maps.shape}, std: {flow_std:.4f}, mean: {np.mean(flow_maps):.4f}")
            
            return frames_np, features_np, flow_maps
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            # Return zero arrays as fallback
            return (np.zeros((self.frame_count, self.image_size, self.image_size, 3), dtype=np.float32),
                    np.zeros((self.frame_count, self.config["model"]["feature_dim"]), dtype=np.float32),
                    np.zeros((self.frame_count-1, self.image_size, self.image_size, 2), dtype=np.float32))