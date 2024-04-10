# processing.py

import json

def parse_json(json_file_path):
    """Load and parse JSON tracking data."""
    with open(json_file_path) as file:
        return json.load(file)
