from __future__ import annotations

import json

from cowbook.io.config_loader import load_config


def test_load_config_applies_defaults_and_normalizes_values(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": "1"}]],
                "fps": "12",
                "output_image_format": "jpeg",
            }
        )
    )

    config = load_config(str(config_path))

    assert config["fps"] == 12
    assert config["output_image_format"] == "jpg"
    assert config["save_tracking_video"] is False
    assert config["create_projection_video"] is True
    assert config["num_tracking_workers"] == 1
    assert config["mask_videos"] is False
    assert config["video_groups"][0][0]["camera_nr"] == 1
    assert config["runtime_root"] == "var"
    assert config["run_name"] == "default"
    assert config["output_root"] == "var/runs/default"
    assert config["output_image_folder"] == "var/runs/default/frames"
    assert config["output_video_folder"] == "var/runs/default/videos"
    assert config["output_json_folder"] == "var/runs/default/json"
    assert config["masked_video_folder"] == "var/cache/masked_videos"
    assert config["masks"]["Ch1"] == "test_img/combined_mask_ch1.png"


def test_load_config_applies_explicit_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]], "fps": 6})
    )

    config = load_config(
        str(config_path),
        overrides={
            "fps": 15,
            "output_image_format": "png",
            "create_projection_video": False,
        },
    )

    assert config["fps"] == 15
    assert config["output_image_format"] == "png"
    assert config["create_projection_video"] is False


def test_load_config_derives_run_scoped_output_paths(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "runtime_root": "runtime",
                "run_name": "experiment-01",
            }
        )
    )

    config = load_config(str(config_path))

    assert config["output_root"] == "runtime/runs/experiment-01"
    assert config["output_image_folder"] == "runtime/runs/experiment-01/frames"
    assert config["output_video_folder"] == "runtime/runs/experiment-01/videos"
    assert config["output_json_folder"] == "runtime/runs/experiment-01/json"
    assert config["masked_video_folder"] == "runtime/cache/masked_videos"


def test_load_config_rejects_duplicate_camera_numbers(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[
                    {"path": "videos/a.mp4", "camera_nr": 1},
                    {"path": "videos/b.mp4", "camera_nr": 1},
                ]]
            }
        )
    )

    assert load_config(str(config_path)) == {}


def test_load_config_rejects_invalid_image_format(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "output_image_format": "bmp",
            }
        )
    )

    assert load_config(str(config_path)) == {}
