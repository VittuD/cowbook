# main.py

import argparse
from config_loader import load_config
from directory_manager import clear_output_directory

def main(config_path):
    config = load_config(config_path)
    
    # Prepare the output directory
    output_image_folder = "output_frames"
    clear_output_directory(output_image_folder)
    
    # Placeholder for processing (in real usage, would iterate over videos)
    print(f"Processing videos with config: {config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and create projection video.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)
