import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from datetime import datetime
import logging

# usage: python3 compute_dsc.py --root_dir '/Users/karissachan/Library/CloudStorage/GoogleDrive-karissa.chan@torontomu.ca/Shared drives/Karissa Chan/NeuroAI Pipeline/wml_segmentation_testing/inference_outputs/RA Models' --ss_dir '/Users/karissachan/Library/CloudStorage/GoogleDrive-karissa.chan@torontomu.ca/Shared drives/Karissa Chan/NeuroAI Pipeline/wml_segmentation_testing/test_inference_data/Task/WML'


# Set up logging configuration
logging.basicConfig(
    filename='dsc_error_logs.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_dsc(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice Similarity Coefficient between prediction and ground truth masks.
    
    Args:
        pred_mask (np.ndarray): Predicted binary mask
        gt_mask (np.ndarray): Ground truth binary mask
    
    Returns:
        float: DSC score between 0 and 1
    """
    try:
        # Ensure masks are binary
        pred_mask = pred_mask.astype(bool)
        gt_mask = gt_mask.astype(bool)
        
        # Calculate intersection and sums
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()
        
        # Compute DSC
        dsc = (2.0 * intersection) / (pred_sum + gt_sum + 1e-6)  # Add small epsilon to avoid division by zero
        
        return float(dsc)
    except Exception as e:
        logging.error(f"Error computing DSC: {str(e)}")
        raise

def process_masks(test_dir: str, ss_dir: str) -> List[Tuple[str, float]]:
    """
    Process all masks in test_dir and ss_dir and compute DSC scores.
    
    Args:
        test_dir (str): Path to directory containing test masks
        ss_dir (str): Path to directory containing silver standard masks
    
    Returns:
        List[Tuple[str, float]]: List of tuples containing (filename, dsc_score)
    """
    results = []
    test_path = Path(test_dir)
    ss_path = Path(ss_dir)
    
    # Get all nifti files in test directory
    test_files = sorted([f for f in test_path.glob('*.nii.gz')])
    
    for test_file in test_files:
        try:
            # Find corresponding file in ss_dir
            ss_file = ss_path / test_file.name
            
            if not ss_file.exists():
                logging.warning(f"No matching silver standard file for {test_file.name}")
                continue
            
            # Load masks
            test_mask = nib.load(test_file).get_fdata()
            ss_mask = nib.load(ss_file).get_fdata()

            # Check if masks contain values other than 0 and 1
            if not np.array_equal(test_mask, test_mask.astype(bool)):
                logging.info(f"{test_file.name} contains non-binary values. Applying threshold at 0.25")
                test_mask = (test_mask >= 0.25).astype(float)
            
            if not np.array_equal(ss_mask, ss_mask.astype(bool)):
                logging.info(f"Silver standard {test_file.name} contains non-binary values. Applying threshold at 0.25")
                ss_mask = (ss_mask >= 0.25).astype(float)

            # ensure masks are all binary
            test_mask = test_mask.astype(bool)
            ss_mask = ss_mask.astype(bool)
            
            # Compute DSC
            dsc_score = compute_dsc(test_mask, ss_mask)
            results.append((test_file.name, dsc_score))
            
        except Exception as e:
            logging.error(f"Error processing {test_file.name}: {str(e)}")
            continue
        
    return results

def main():
    """
    Main function to compute DSC scores for WML masks across multiple model outputs.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute DSC scores for WML masks')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing multiple test folders')
    parser.add_argument('--ss_dir', type=str, required=True,
                      help='Directory containing silver standard masks')
    
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    all_results = []
    
    # Process each test folder (model output) in the root directory
    for test_folder in sorted(root_path.iterdir()):
        if not test_folder.is_dir():
            continue
            
        # Look for wml_masks subfolder
        wml_masks_dir = test_folder / 'wml_masks'
        if not wml_masks_dir.exists() or not wml_masks_dir.is_dir():
            print(f"Warning: No wml_masks folder found in {test_folder.name}, skipping...")
            continue
            
        model_type = test_folder.name
        print(f"\nProcessing model: {model_type}")
        
        # Get DSC scores for this model using the wml_masks subfolder
        results = process_masks(str(wml_masks_dir), args.ss_dir)
        
        # Add model type to results
        model_results = [(filename, score, model_type) for filename, score in results]
        all_results.extend(model_results)
    
    if not all_results:
        print("No results found. Check if the test folders contain wml_masks directories with valid mask files.")
        return
        
    # Create DataFrame for all results
    df_results = pd.DataFrame(all_results, columns=['Filename', 'DSC', 'Model'])
    
    # Calculate summary statistics per model
    summary_stats = df_results.groupby('Model').agg({
        'DSC': ['mean', 'std', 'count']
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['Mean DSC', 'Std DSC', 'Number of Cases']
    summary_stats = summary_stats.reset_index()
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.root_dir, f'dsc_scores_comparison_{timestamp}.xlsx')
    
    # Create Excel writer object with both sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Individual Scores', index=False)
        summary_stats.to_excel(writer, sheet_name='Summary by Model', index=False)
    
    # Print summary to console
    print("\nResults Summary by Model:")
    print(summary_stats.to_string(index=False))
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
