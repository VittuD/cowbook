# Project Title

This project is focused on video processing and image analysis, specifically for tracking and projection tasks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [License](#license)

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

To run the main script, use:

```sh
python main.py config.json
```

This will load the configuration from `config.json` and process the videos as specified.

## Configuration

The configuration file `config.json` contains various settings for the project:

```json
{
    "model_path": "models/best.pt",
    "calibration_file": "legacy/calibration_matrix.json",
    "video_paths": [
        ["videos/block_0.mp4", 1]
    ],
    "save_tracking_video": true,
    "create_projection_video": true,
    "fps": 6
}
```

- `model_path`: Path to the model file.
- `calibration_file`: Path to the calibration file.
- `video_paths`: List of video files and their corresponding IDs.
- `save_tracking_video`: Boolean to save the tracking video.
- `create_projection_video`: Boolean to create the projection video.
- `fps`: Frames per second for the output video.

# Directory Structure

## Directory Structure

The directory structure of the project is as follows:

```
├── .gitignore
├── config_loader.py
├── config.json
├── directory_manager.py
├── frame_processor.py
├── legacy/
│   ├── __init__.py
│   ├── calibration_matrix.json
│   ├── image_utils.py
│   ├── points_data.py
│   └── real_world_points.json
├── main.py
├── models/
│   ├── .gitkeep
│   └── best.pt
├── output_frames/
├── processing.py
├── readme.md
├── requirements.txt
├── tracking.py
├── video_generator.py
├── video_processor.py
└── videos/
    └── block_0_tracking.json
```

## License

This project is licensed under the MIT License.