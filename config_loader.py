# config_loader.py

import json

def load_config(config_path):
    """
    Load configuration settings from a JSON file.
    """
    with open(config_path) as f:
        return json.load(f)
