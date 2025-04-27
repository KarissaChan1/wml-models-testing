import os
import numpy as np
import pandas as pd
import nibabel as nib
import random
import shutil
import argparse

# SAMPLE TEST VOLUMES FROM ORIGINAL NEURO DB VOLUMES
# usage: python3 sample_test_volumes.py --data_path "/Users/karissachan/Library/CloudStorage/GoogleDrive-karissa.chan@torontomu.ca/Shared drives/Karissa Chan/NeuroAI Pipeline/Neuro_DB" --out_path "/Users/karissachan/Library/CloudStorage/GoogleDrive-karissa.chan@torontomu.ca/Shared drives/Karissa Chan/NeuroAI Pipeline/wml_segmentation_testing/test_inference_data" --dataset_names ADNI CAIN ONDRI CCNA --n_vols 5

def sample_test_volumes(data_path, dataset_names, out_path, n_vols, standardization_folders):
    """
    Samples test volumes from each dataset and saves them to the output directory.
    
    Args:
        data_path (str): Path to the dataset directory
        out_path (str): Path to the output directory
        dataset_names (list): List of dataset names to sample from
        n_vols (int): Number of volumes to sample from each dataset
        standardization_folders (list): List of standardization folder names (e.g., ['V2', 'V4'])
    """

    sampled_data = []
    random.seed(42)

    # Create output directories for each standardization folder and mask
    out_paths = {}
    for std_folder in standardization_folders:
        std_path = os.path.join(out_path, f'Standardized/{std_folder}')
        os.makedirs(std_path, exist_ok=True)
        out_paths[std_folder] = std_path
    # Mask output
    mask_out_path = os.path.join(out_path, 'Task/WML')
    os.makedirs(mask_out_path, exist_ok=True)
    out_paths['Task/WML'] = mask_out_path

    # Sample volumes
    for dataset_name in dataset_names:
        print(f"Sampling {n_vols} volumes from dataset: {dataset_name}")
        try:
            # Build source paths for each standardization folder
            vol_paths = {}
            for std_folder in standardization_folders:
                vol_path = os.path.join(data_path, f'Standardized/{std_folder}/{dataset_name}')
                # Special case for ADNI and V2
                if dataset_name == 'ADNI' and std_folder == 'V2':
                    vol_path = os.path.join(vol_path, 'standardized')
                vol_paths[std_folder] = vol_path
            # Mask path
            mask_path = os.path.join(data_path, f'Task/WML/{dataset_name}')

            # Check if all required directories exist
            all_dirs = list(vol_paths.values()) + [mask_path]
            if not all(os.path.exists(path) for path in all_dirs):
                print(f"Warning: One or more required directories missing for dataset {dataset_name}, skipping...")
                continue

            # Get list of volumes for the current dataset (from the first standardization folder)
            try:
                first_std_folder = standardization_folders[0]
                vol_list = os.listdir(vol_paths[first_std_folder])
                if not vol_list:
                    print(f"Warning: No volumes found in {vol_paths[first_std_folder]}, skipping dataset {dataset_name}")
                    continue

                if len(vol_list) < n_vols:
                    print(f"Warning: Dataset {dataset_name} has fewer volumes ({len(vol_list)}) than requested ({n_vols}), adjusting sample size...")
                    n_vols = len(vol_list)

            except PermissionError:
                print(f"Error: No permission to access directory for dataset {dataset_name}, skipping...")
                continue
            except Exception as e:
                print(f"Error accessing volumes for dataset {dataset_name}: {str(e)}, skipping...")
                continue

            # Sample n_vols volumes
            sampled_vols = random.sample(vol_list, n_vols)

            # Copy sampled volumes to output directory
            for vol in sampled_vols:
                try:
                    # Track which files to process
                    paths_to_process = []
                    out_files = {}
                    src_files = {}

                    # For each standardization folder
                    for std_folder in standardization_folders:
                        out_file = os.path.join(out_paths[std_folder], vol)
                        src_file = os.path.join(vol_paths[std_folder], vol)
                        out_files[std_folder] = out_file
                        src_files[std_folder] = src_file
                        if not os.path.exists(out_file):
                            paths_to_process.append((src_file, out_file, std_folder))
                        else:
                            print(f"Warning: Volume {vol} already exists in {std_folder} output directory, skipping {std_folder}...")

                    # Mask file
                    out_mask_file = os.path.join(out_paths['Task/WML'], vol)
                    mask_src = os.path.join(mask_path, vol)
                    out_files['mask'] = out_mask_file
                    src_files['mask'] = mask_src
                    if not os.path.exists(out_mask_file):
                        paths_to_process.append((mask_src, out_mask_file, 'mask'))
                    else:
                        print(f"Warning: Volume {vol} already exists in mask output directory, skipping mask...")

                    if not paths_to_process:
                        print(f"All outputs already exist for volume {vol}, skipping entirely...")
                        continue

                    # Check if all source files exist
                    if not all(os.path.isfile(src) for src, _, _ in paths_to_process):
                        print(f"Warning: Missing one or more source files for volume {vol} in dataset {dataset_name}, skipping...")
                        continue

                    # Copy files
                    for src, dst, folder in paths_to_process:
                        shutil.copy(src, dst)

                    # Record sampled data
                    record = {
                        'dataset_name': dataset_name,
                        'vol_name': vol,
                        'mask_path': out_mask_file
                    }
                    for std_folder in standardization_folders:
                        record[f'vol_path_{std_folder}'] = out_files[std_folder]
                    sampled_data.append(record)

                except PermissionError:
                    print(f"Error: Permission denied while copying volume {vol} from dataset {dataset_name}, skipping...")
                    continue
                except Exception as e:
                    print(f"Error copying volume {vol} from dataset {dataset_name}: {str(e)}, skipping...")
                    continue

        except Exception as e:
            print(f"Unexpected error processing dataset {dataset_name}: {str(e)}, skipping...")
            continue
    
    # Save sampled data to Excel file if any volumes were successfully sampled
    if sampled_data:
        try:
            sampled_data_df = pd.DataFrame(sampled_data)
            excel_path = os.path.join(out_path, 'sampled_data.xlsx')
            sampled_data_df.to_excel(excel_path, index=False)
            print(f"Successfully saved sampling information to {excel_path}")
        except Exception as e:
            print(f"Error saving sampling information to Excel: {str(e)}")
    else:
        print("Warning: No volumes were successfully sampled")
    
    return sampled_data

def find_matching_file(directory, base_filename):
    """
    Find a file in the directory that matches the base filename, regardless of .nii or .nii.gz extension
    
    Args:
        directory (str): Directory to search in
        base_filename (str): Base filename to match (without extension)
    
    Returns:
        str: Full matching filename if found, None otherwise
    """
    # Remove .nii or .nii.gz extension if present
    if base_filename.endswith('.nii.gz'):
        base_filename = base_filename[:-7]
    elif base_filename.endswith('.nii'):
        base_filename = base_filename[:-4]
    
    try:
        for filename in os.listdir(directory):
            current_base = filename
            if current_base.endswith('.nii.gz'):
                current_base = current_base[:-7]
            elif current_base.endswith('.nii'):
                current_base = current_base[:-4]
            
            if current_base == base_filename:
                return filename
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing directory {directory}: {str(e)}")
    return None

def sample_original_volumes(out_path, excel_path):
    """
    Samples original volumes using the excel file output from the previous function that contains the list of volumes to sample.

    Args:
        out_path (str): Path to the output directory
        excel_path (str): Path to the excel file containing the list of volumes to sample
    """

    original_data_path = "/Users/karissachan/Library/CloudStorage/GoogleDrive-karissa.chan@torontomu.ca/.shortcut-targets-by-id/1viatHUAHD50HMOlRolRwJpIe5FbFBz62/Neuro_DB/Original_Reoriented"
    print(f"Sampling original volumes from {original_data_path}")
    dest_path = os.path.join(out_path, 'Original')
    os.makedirs(dest_path, exist_ok=True)

    sampled_data = pd.read_excel(excel_path)
    for index, row in sampled_data.iterrows():
        dataset_name = row['dataset_name']
        vol_name = row['vol_name']

        vol_path = os.path.join(original_data_path, f'{dataset_name}_FLAIR')

        matching_file = find_matching_file(vol_path, vol_name)

        if matching_file:
            print(f"Found matching file: {matching_file}")
            source_file = os.path.join(vol_path, matching_file)
            
            try:
                # If it's a .nii file, load and save as .nii.gz
                if matching_file.endswith('.nii'):
                    print(f"Converting {matching_file} to .nii.gz format")
                    img = nib.load(source_file)
                    gz_filename = matching_file + '.gz'
                    dest_file = os.path.join(dest_path, gz_filename)
                    nib.save(img, dest_file)
                else:
                    # If it's already .nii.gz, just copy
                    dest_file = os.path.join(dest_path, matching_file)
                    shutil.copy(source_file, dest_file)
            except Exception as e:
                print(f"Error processing file {matching_file}: {str(e)}")
                continue
        else:
            print(f"No matching file found for {vol_name}")
            continue

    print(f"Successfully sampled {len(sampled_data)} original volumes")

def main():
    parser = argparse.ArgumentParser(description='Sample test volumes from datasets')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save the sampled volumes')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True, help='List of dataset names to sample from')
    # ['ADNI','CAIN','CCNA','ONDRI']
    parser.add_argument('--n_vols', type=int, default=5, help='Number of volumes to sample from each dataset')
    parser.add_argument('--original_vols', type=bool, default=True, help='Whether to sample original volumes')
    parser.add_argument('--std_folders', type=str, nargs='+', default=['V2', 'V4'],
                        help="List of standardization folders to use (default: ['V2', 'V4'])")
    args = parser.parse_args()
    
    sample_test_volumes(
        args.data_path,
        args.dataset_names,
        args.out_path,
        args.n_vols,
        args.std_folders
    )
    if args.original_vols:
        sample_original_volumes(args.out_path, os.path.join(args.out_path, 'sampled_data.xlsx'))

if __name__ == '__main__':
    main()
