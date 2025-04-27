import logging
import yaml
import os
from datetime import datetime

# usage: python3 get_inference_commands.py
def main():# Load configuration
    try:
        with open('model_inference_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return

    # Extract paths from config
    input_volume_dir = config['volume_input_dir']
    model_root_path = config['model_root_path']
    output_base_dir = config['output_dir']

     # Create commands list
    commands = []
    
    # Add initial setup commands
    commands.append("# First, run these commands in your terminal:")
    commands.append("cd /Users/karissachan/Documents/GitHub/wml-segmentation")
    commands.append("poetry env use 3.10.11")
    commands.append("poetry shell")
    commands.append("If need to re-generate silver standard SCUNET WML masks, run the following command:")
    commands.append(f"wml_inference -v {input_volume_dir} -m ./models/model_scunet_GDL.hdf5 -c scunet -s '/Users/karissachan/Library/CloudStorage/GoogleDrive-karissa.chan@torontomu.ca/Shared drives/Karissa Chan/NeuroAI Pipeline/wml_segmentation_testing/test_inference_data/Task/WML_Reoriented'")
    commands.append("\n# Then run each of these inference commands:")
    
    # Loop through models and run inference
    for model in config['models']:
        model_name = model['name']
        model_rel_path = model['path']
        model_full_path = os.path.join(model_root_path, model_rel_path)
        save_dir = os.path.join(output_base_dir, model_name)
        
        # Determine model choice based on path
        model_choice = "scunet"
        if "sc-unet" not in model_full_path.lower() or "scunet" not in model_full_path.lower():
            model_choice = "unet"
        
        inference_command = (
            f"wml_inference "
            f"-v '{input_volume_dir}' "
            f"-m '{model_full_path}' "
            f"-c {model_choice} "
            f"-s '{save_dir}'"
        )

        commands.append(inference_command)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    output_file = f'inference_commands_{timestamp}.txt'
    
    # Write commands to file
    with open(output_file, 'w') as f:
        f.write('\n\n'.join(commands))
    
    print(f"Commands have been saved to: {output_file}")

if __name__ == "__main__":
    main()