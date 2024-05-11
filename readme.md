# Project Title

This project is focused on video processing and image analysis, specifically for tracking and projection tasks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
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

## License

This project is licensed under the MIT License.