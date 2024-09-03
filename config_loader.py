# config_loader.py

import json

def load_config(config_path):
    """
    Load configuration settings from a JSON file with error handling and default values.
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
        # Provide default values for missing config options
        config.setdefault("model_path", "models/best.pt")
        config.setdefault("fps", 6)
        config.setdefault("save_tracking_video", False)
        config.setdefault("create_projection_video", True)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Configuration file {config_path} is not valid JSON.")
        return {}
