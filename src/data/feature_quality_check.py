import os
import numpy as np
from pathlib import Path
import yaml
import logging

def get_config():
    # Navigate to the project root and load the configuration file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Set up logging with timestamp and level information
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration and extract relevant parameters
    config = get_config()
    feature_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), config['data']['feature_dir']))
    frame_count = config['data']['frame_count']
    image_size = config['data']['image_size']
    feature_dim = config['model']['feature_dim']

    # Define expected feature types in each NPZ file
    all_keys = ['frames', 'appearance', 'flow']

    # Initialize counters for tracking validation results
    corrupted = []  # List to store files with issues
    passed = 0      # Counter for files that pass all checks

    # Find all NPZ feature files in the specified directory
    feature_files = list(feature_dir.glob('*.npz'))
    
    # Exit early if no files are found
    if not feature_files:
        logger.warning(f"No feature files found in {feature_dir}")
        return

    logger.info(f"Starting feature quality check for {len(feature_files)} files...")

    # Examine each feature file for quality issues
    for npz_file in feature_files:
        file_issues = []  # Track issues found in this file
        try:
            # Load the feature file
            features = np.load(npz_file)
            logger.info(f"Checking file: {npz_file.name}")
            
            # Check each expected feature type
            for key in all_keys:
                # Verify the key exists in the file
                if key not in features:
                    logger.error(f"Missing key: {key}")
                    file_issues.append(f"missing {key}")
                    continue
                    
                # Log basic statistics for the feature data
                data = features[key]
                logger.info(f"  {key}: shape={data.shape}, dtype={data.dtype}, min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}, std={np.std(data):.4f}")
                
                # Check for NaN values (invalid data)
                if np.isnan(data).any():
                    logger.warning(f"  {key} contains NaNs!")
                    file_issues.append(f"NaN in {key}")
                    
                # Check for all-zero arrays (failed extraction)
                if np.all(data == 0):
                    logger.warning(f"  {key} is all zeros!")
                    file_issues.append(f"all zeros in {key}")
                    
                # Check for low variance (poor quality features)
                if np.abs(np.mean(data)) < config['data']['low_variance_threshold'] and np.std(data) < config['data']['low_variance_threshold']:
                    logger.warning(f"  {key} may contain zeroed or low-variance data")
                    file_issues.append(f"low variance in {key}")
            
            # Verify dimensions match expected shapes
            if 'frames' in features and features['frames'].shape != (frame_count, image_size, image_size, 3):
                logger.error(f"  frames shape {features['frames'].shape} (expected {(frame_count, image_size, image_size, 3)})")
                file_issues.append(f"frames shape {features['frames'].shape}")
                
            if 'appearance' in features and features['appearance'].shape != (frame_count, feature_dim):
                logger.error(f"  appearance shape {features['appearance'].shape} (expected {(frame_count, feature_dim)})")
                file_issues.append(f"appearance shape {features['appearance'].shape}")
                
            if 'flow' in features and features['flow'].shape != (frame_count-1, image_size, image_size, 2):
                logger.error(f"  flow shape {features['flow'].shape} (expected {(frame_count-1, image_size, image_size, 2)})")
                file_issues.append(f"flow shape {features['flow'].shape}")
            
            # Track results based on whether issues were found
            if file_issues:
                corrupted.append((npz_file, file_issues))
                logger.warning(f"  Issues found: {file_issues}")
            else:
                passed += 1
                logger.info(f"  ✓ All checks passed")
                
        except Exception as e:
            # Handle file loading errors
            logger.error(f"  Error loading {npz_file}: {e}")
            corrupted.append((npz_file, [f'Load error: {e}']))

    # Generate summary report of validation results
    logger.info("=" * 50)
    logger.info("FEATURE QUALITY CHECK SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Passed all checks: {passed}")
    logger.info(f"Files with issues: {len(corrupted)}")
    
    # List details of problematic files if any were found
    if corrupted:
        logger.warning("Files with issues:")
        for f, issues in corrupted:
            logger.warning(f"  {f.name}: {issues}")
    
    logger.info(f"Total feature files checked: {len(feature_files)}")
    
    if passed == len(feature_files):
        logger.info("🎉 All feature files passed quality checks!")
    else:
        logger.warning(f"⚠️  {len(corrupted)} files have issues that need attention")

if __name__ == "__main__":
    main()