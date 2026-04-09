from __future__ import annotations

import json

import pytest

from cowbook.io.config_loader import (
    load_config_file,
    normalize_config_mapping,
    write_config_file,
)


def test_load_config_file_applies_defaults_and_normalizes_values(tmp_path):
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

    config = load_config_file(str(config_path))

    assert config["fps"] == 12
    assert config["output_image_format"] == "jpg"
    assert config["save_tracking_video"] is False
    assert config["create_projection_video"] is True
    assert config["tracking_concurrency"] == 1
    assert config["log_progress"] is False
    assert config["mask_videos"] is False
    assert config["video_groups"][0][0]["camera_nr"] == 1
    assert config["runtime_root"] == "var"
    assert config["run_name"] == "default"
    assert config["output_root"] == "var/runs/default"
    assert config["output_image_folder"] == "var/runs/default/frames"
    assert config["output_video_folder"] == "var/runs/default/videos"
    assert config["output_json_folder"] == "var/runs/default/json"
    assert config["masked_video_folder"] == "var/cache/masked_videos"
    assert config["masks"]["Ch1"] == "assets/masks/combined_mask_ch1.png"
    assert config["tracking_cleanup"]["enabled"] is False
    assert config["tracking_cleanup"]["nms_mode"] == "hybrid_nms"


def test_normalize_config_mapping_applies_defaults_and_overrides():
    config = normalize_config_mapping(
        {
            "video_groups": [[{"path": "videos/example.mp4", "camera_nr": "1"}]],
            "fps": "12",
        },
        overrides={"run_name": "normalized"},
    )

    assert config["fps"] == 12
    assert config["run_name"] == "normalized"
    assert config["output_root"] == "var/runs/normalized"
    assert config["video_groups"][0][0]["camera_nr"] == 1
    assert config["tracking_cleanup"]["min_track_total_observations"] is None
    assert config["tracking_cleanup"]["min_area_ratio"] is None
    assert config["tracking_cleanup"]["min_mask_fill_ratio"] is None
    assert config["tracking_cleanup"]["postprocess_gap_fill"] is False


def test_load_config_file_applies_explicit_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]], "fps": 6})
    )

    config = load_config_file(
        str(config_path),
        overrides={
            "fps": 15,
            "output_image_format": "png",
            "create_projection_video": False,
            "log_progress": True,
        },
    )

    assert config["fps"] == 15
    assert config["output_image_format"] == "png"
    assert config["create_projection_video"] is False
    assert config["log_progress"] is True


def test_load_config_file_preserves_loader_specific_errors(tmp_path):
    missing_path = tmp_path / "missing.json"
    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{bad json")

    with pytest.raises(FileNotFoundError):
        load_config_file(str(missing_path))

    with pytest.raises(json.JSONDecodeError):
        load_config_file(str(invalid_json_path))


def test_load_config_file_derives_run_scoped_output_paths(tmp_path):
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

    config = load_config_file(str(config_path))

    assert config["output_root"] == "runtime/runs/experiment-01"
    assert config["output_image_folder"] == "runtime/runs/experiment-01/frames"
    assert config["output_video_folder"] == "runtime/runs/experiment-01/videos"
    assert config["output_json_folder"] == "runtime/runs/experiment-01/json"
    assert config["masked_video_folder"] == "runtime/cache/masked_videos"


def test_write_config_file_materializes_normalized_config(tmp_path):
    output_path = tmp_path / "nested" / "materialized.json"

    written_path = write_config_file(
        {
            "video_groups": [[{"path": "videos/example.mp4", "camera_nr": "1"}]],
            "fps": "12",
        },
        str(output_path),
        overrides={"run_name": "materialized"},
    )

    assert written_path == str(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["fps"] == 12
    assert saved["run_name"] == "materialized"
    assert saved["output_root"] == "var/runs/materialized"
    assert saved["video_groups"][0][0]["camera_nr"] == 1


def test_load_config_file_rejects_duplicate_camera_numbers(tmp_path):
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

    with pytest.raises(ValueError, match="associated with a unique camera"):
        load_config_file(str(config_path))


def test_load_config_file_rejects_invalid_image_format(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "output_image_format": "bmp",
            }
        )
    )

    with pytest.raises(ValueError, match="Invalid output_image_format"):
        load_config_file(str(config_path))


def test_load_config_file_rejects_non_positive_tracking_concurrency(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "tracking_concurrency": 0,
            }
        )
    )

    with pytest.raises(ValueError, match="'tracking_concurrency' must be >= 1"):
        load_config_file(str(config_path))


def test_load_config_file_normalizes_tracking_cleanup_block(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "tracking_cleanup": {
                    "enabled": True,
                    "roi": [[0, 0], [10, 0], [10, 10]],
                    "min_track_length": 5,
                    "min_area_ratio": 0.02,
                    "min_mask_fill_ratio": 0.15,
                    "postprocess_gap_fill": True,
                },
            }
        )
    )

    config = load_config_file(str(config_path))

    assert config["tracking_cleanup"]["enabled"] is True
    assert config["tracking_cleanup"]["roi"] == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    assert config["tracking_cleanup"]["min_track_length"] == 5
    assert config["tracking_cleanup"]["min_area_ratio"] == pytest.approx(0.02)
    assert config["tracking_cleanup"]["min_mask_fill_ratio"] == pytest.approx(0.15)
    assert config["tracking_cleanup"]["postprocess_gap_fill"] is True


def test_load_config_file_rejects_invalid_tracking_cleanup_nms_mode(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "tracking_cleanup": {"nms_mode": "bad_mode"},
            }
        )
    )

    with pytest.raises(ValueError, match="tracking_cleanup.nms_mode"):
        load_config_file(str(config_path))


def test_load_config_file_rejects_tracking_cleanup_bad_roi(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "tracking_cleanup": {"roi": [[0, 0], [10, 0]]},
            }
        )
    )

    with pytest.raises(ValueError, match="tracking_cleanup.roi must be a polygon"):
        load_config_file(str(config_path))


@pytest.mark.parametrize(
    ("tracking_cleanup", "message"),
    [
        ({"smoothing_alpha": 1.0}, "tracking_cleanup.smoothing_alpha"),
        ({"min_area_px": 5, "max_area_px": 3}, "tracking_cleanup.min_area_px"),
        ({"min_area_ratio": 0.4, "max_area_ratio": 0.3}, "tracking_cleanup.min_area_ratio"),
        ({"min_area_ratio": -0.1}, "tracking_cleanup.min_area_ratio"),
        ({"max_area_ratio": 1.1}, "tracking_cleanup.max_area_ratio"),
        ({"min_mask_fill_ratio": -0.1}, "tracking_cleanup.min_mask_fill_ratio"),
        ({"min_mask_fill_ratio": 1.1}, "tracking_cleanup.min_mask_fill_ratio"),
        ({"min_aspect_ratio": 2.0, "max_aspect_ratio": 1.0}, "tracking_cleanup.min_aspect_ratio"),
        ({"edge_margin_px": -1}, "tracking_cleanup.edge_margin_px"),
        ({"gap_fill_max_frames": -1}, "tracking_cleanup.gap_fill_max_frames"),
        ({"max_center_speed_px_per_frame": -1}, "tracking_cleanup.max_center_speed_px_per_frame"),
        ({"max_relative_area_change": -1}, "tracking_cleanup.max_relative_area_change"),
        ({"max_relative_aspect_change": -1}, "tracking_cleanup.max_relative_aspect_change"),
    ],
)
def test_normalize_config_mapping_rejects_invalid_tracking_cleanup_values(tracking_cleanup, message):
    with pytest.raises(ValueError, match=message):
        normalize_config_mapping(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "tracking_cleanup": tracking_cleanup,
            }
        )


def test_normalize_config_mapping_rejects_invalid_shape_errors():
    with pytest.raises(ValueError, match="'fps' must be an integer"):
        normalize_config_mapping({"video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]], "fps": "bad"})

    with pytest.raises(ValueError, match="'video_groups' must be a list"):
        normalize_config_mapping({"video_groups": "bad"})

    with pytest.raises(ValueError, match="Group 1 must be a list"):
        normalize_config_mapping({"video_groups": ["bad"]})

    with pytest.raises(ValueError, match="between 1 and 4 videos"):
        normalize_config_mapping({"video_groups": [[]]})

    with pytest.raises(ValueError, match="between 1 and 4 videos"):
        normalize_config_mapping(
            {
                "video_groups": [[
                    {"path": "a.mp4", "camera_nr": 1},
                    {"path": "b.mp4", "camera_nr": 2},
                    {"path": "c.mp4", "camera_nr": 3},
                    {"path": "d.mp4", "camera_nr": 4},
                    {"path": "e.mp4", "camera_nr": 5},
                ]]
            }
        )

    with pytest.raises(ValueError, match="missing 'path' or 'camera_nr'"):
        normalize_config_mapping({"video_groups": [[{"path": "a.mp4"}]]})

    with pytest.raises(ValueError, match="non-integer camera_nr"):
        normalize_config_mapping({"video_groups": [[{"path": "a.mp4", "camera_nr": "bad"}]]})


def test_normalize_config_mapping_does_not_mutate_nested_input():
    config = {
        "video_groups": [[{"path": "videos/example.mp4", "camera_nr": "1"}]],
        "tracking_cleanup": {"roi": [[0, "1"], ["2", 3], [4, 5]]},
    }

    normalized = normalize_config_mapping(config)

    assert normalized["video_groups"][0][0]["camera_nr"] == 1
    assert normalized["tracking_cleanup"]["roi"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert config["video_groups"][0][0]["camera_nr"] == "1"
    assert config["tracking_cleanup"]["roi"] == [[0, "1"], ["2", 3], [4, 5]]


def test_write_config_file_uses_same_validation_contract(tmp_path):
    output_path = tmp_path / "materialized.json"

    with pytest.raises(ValueError, match="Invalid output_image_format"):
        write_config_file(
            {
                "video_groups": [[{"path": "videos/example.mp4", "camera_nr": 1}]],
                "output_image_format": "bmp",
            },
            str(output_path),
        )
