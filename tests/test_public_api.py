from __future__ import annotations

import json

import cowbook
from cowbook import runtime


def test_package_root_exposes_public_runtime_surface():
    assert cowbook.PipelineRunner is runtime.PipelineRunner
    assert cowbook.PipelineConfig is runtime.PipelineConfig
    assert callable(cowbook.run_pipeline)
    assert callable(cowbook.load_pipeline_config)
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


def test_run_pipeline_helper_delegates_to_runner(monkeypatch):
    captured = {}

    class FakeRunner:
        def run(self, config_path, overrides=None, **kwargs):
            captured["config_path"] = config_path
            captured["overrides"] = overrides
            captured["kwargs"] = kwargs
            return {"ok": True}

    result = runtime.run_pipeline(
        "config.json",
        overrides={"fps": 9},
        job_id="job-1",
        runner=FakeRunner(),
    )

    assert result == {"ok": True}
    assert captured["config_path"] == "config.json"
    assert captured["overrides"] == {"fps": 9}
    assert captured["kwargs"]["job_id"] == "job-1"
