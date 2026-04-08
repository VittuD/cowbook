from __future__ import annotations

import json

import cowbook
import pytest
from cowbook import runtime


def test_package_root_exposes_public_runtime_surface():
    assert cowbook.PipelineRunner is runtime.PipelineRunner
    assert cowbook.PipelineConfig is runtime.PipelineConfig
    assert cowbook.RunRequest is runtime.RunRequest
    assert cowbook.RunResult is runtime.RunResult
    assert callable(cowbook.run_pipeline)
    assert callable(cowbook.run_pipeline_request)
    assert callable(cowbook.load_pipeline_config)
    assert callable(cowbook.load_pipeline_config_object)
    assert callable(cowbook.materialize_pipeline_config)
    assert not hasattr(cowbook, "JobManagerService")
    assert not hasattr(cowbook, "create_job_manager")


def test_load_pipeline_config_returns_typed_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "models/test.pt",
                "fps": 7,
                "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
            }
        )
    )

    config = runtime.load_pipeline_config(str(config_path), overrides={"run_name": "runtime_test"})

    assert isinstance(config, runtime.PipelineConfig)
    assert config.fps == 7
    assert config.run_name == "runtime_test"
    assert config.video_groups[0][0].camera_nr == 1


def test_load_pipeline_config_object_returns_typed_config():
    config = runtime.load_pipeline_config_object(
        {
            "model_path": "models/test.pt",
            "fps": 7,
            "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
        },
        overrides={"run_name": "runtime_object_test"},
    )

    assert isinstance(config, runtime.PipelineConfig)
    assert config.fps == 7
    assert config.run_name == "runtime_object_test"
    assert config.video_groups[0][0].camera_nr == 1


def test_runtime_config_helpers_share_validation_errors(tmp_path):
    config_path = tmp_path / "config.json"
    invalid_config = {
        "model_path": "models/test.pt",
        "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
        "output_image_format": "bmp",
    }
    config_path.write_text(json.dumps(invalid_config))

    with pytest.raises(ValueError, match="Invalid output_image_format"):
        runtime.load_pipeline_config(str(config_path))

    with pytest.raises(ValueError, match="Invalid output_image_format"):
        runtime.load_pipeline_config_object(invalid_config)


def test_load_pipeline_config_preserves_file_loader_errors(tmp_path):
    missing_path = tmp_path / "missing.json"
    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{bad json")

    with pytest.raises(FileNotFoundError):
        runtime.load_pipeline_config(str(missing_path))

    with pytest.raises(json.JSONDecodeError):
        runtime.load_pipeline_config(str(invalid_json_path))


def test_load_pipeline_config_object_does_not_mutate_nested_input():
    config_input = {
        "model_path": "models/test.pt",
        "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": "1"}]],
        "tracking_cleanup": {"roi": [[0, "1"], ["2", 3], [4, 5]]},
    }

    config = runtime.load_pipeline_config_object(config_input)

    assert config.video_groups[0][0].camera_nr == 1
    assert config.tracking_cleanup.roi == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert config_input["video_groups"][0][0]["camera_nr"] == "1"
    assert config_input["tracking_cleanup"]["roi"] == [[0, "1"], ["2", 3], [4, 5]]


def test_run_pipeline_helper_delegates_to_runner(monkeypatch):
    captured = {}

    class FakeRunner:
        def run_request(self, request, **kwargs):
            captured["request"] = request
            captured["kwargs"] = kwargs
            return runtime.RunResult(
                job_run=runtime.JobRun(job_id="job-1", config_path="config.json"),
            )

    observer = object()
    cancellation_token = runtime.CancellationToken()

    result = runtime.run_pipeline(
        "config.json",
        overrides={"fps": 9},
        job_id="job-1",
        observer=observer,
        cancellation_token=cancellation_token,
        runner=FakeRunner(),
    )

    assert isinstance(result, runtime.RunResult)
    assert captured["request"] == runtime.RunRequest(
        config_path="config.json",
        overrides={"fps": 9},
    )
    assert captured["kwargs"]["job_id"] == "job-1"
    assert captured["kwargs"]["observer"] is observer
    assert captured["kwargs"]["cancellation_token"] is cancellation_token


def test_run_pipeline_request_helper_delegates_to_runner():
    captured = {}

    class FakeRunner:
        def run_request(self, request, **kwargs):
            captured["request"] = request
            captured["kwargs"] = kwargs
            return runtime.RunResult(
                job_run=runtime.JobRun(job_id="job-2", config_path="<in-memory>"),
            )

    observer = object()
    cancellation_token = runtime.CancellationToken()

    request = runtime.RunRequest(
        config={
            "model_path": "models/test.pt",
            "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
        }
    )

    result = runtime.run_pipeline_request(
        request,
        job_id="job-2",
        observer=observer,
        cancellation_token=cancellation_token,
        runner=FakeRunner(),
    )

    assert isinstance(result, runtime.RunResult)
    assert captured["request"] == request
    assert captured["kwargs"]["job_id"] == "job-2"
    assert captured["kwargs"]["observer"] is observer
    assert captured["kwargs"]["cancellation_token"] is cancellation_token


def test_materialize_pipeline_config_writes_normalized_config(tmp_path):
    output_path = tmp_path / "materialized.json"

    written_path = runtime.materialize_pipeline_config(
        {
            "model_path": "models/test.pt",
            "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": "1"}]],
        },
        str(output_path),
        overrides={"run_name": "runtime_materialized"},
    )

    assert written_path == str(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["run_name"] == "runtime_materialized"
    assert saved["output_root"] == "var/runs/runtime_materialized"
    assert saved["video_groups"][0][0]["camera_nr"] == 1


def test_materialize_pipeline_config_uses_same_validation_contract(tmp_path):
    output_path = tmp_path / "materialized.json"

    with pytest.raises(ValueError, match="Invalid output_image_format"):
        runtime.materialize_pipeline_config(
            {
                "model_path": "models/test.pt",
                "video_groups": [[{"path": "sample_data/videos/Ch1_60.mp4", "camera_nr": 1}]],
                "output_image_format": "bmp",
            },
            str(output_path),
        )
