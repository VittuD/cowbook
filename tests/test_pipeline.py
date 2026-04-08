from __future__ import annotations

import pytest

from cowbook.app.pipeline import PipelineRunner
from cowbook.core.contracts import PipelineConfig, RunRequest
from cowbook.execution import CancellationToken, InMemoryJobStore, JobCancelledError


class FakeConfigService:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.normalize_calls = []

    def load(self, config_path, overrides=None):
        self.calls.append((config_path, overrides))
        return self.config

    def normalize(self, config, overrides=None):
        self.normalize_calls.append((config, overrides))
        return self.config


class FakeDirectoryService:
    def __init__(self):
        self.prepared = []
        self.cleared = []

    def prepare_output_dirs(self, config):
        self.prepared.append(config)
        return ("frames", "videos", "json", "masked")

    def clear_output_directory(self, directory_path):
        self.cleared.append(directory_path)


class FakeMaskingService:
    def __init__(self, groups):
        self.groups = groups
        self.calls = []

    def preprocess(self, config, **kwargs):
        self.calls.append((config, kwargs))
        return self.groups


class FakeGroupProcessingService:
    def __init__(self):
        self.calls = []

    def process_group(self, *args, reporter=None, cancellation_token=None):
        self.calls.append((args, cancellation_token))
        if reporter is not None:
            reporter.emit("group_completed", stage="group", group_idx=args[0])


class FakeVideoService:
    def __init__(self):
        self.calls = []

    def create_projection_video(self, image_folder, output_video_path, fps, **kwargs):
        self.calls.append((image_folder, output_video_path, fps, kwargs))
        if getattr(self, "should_fail", False):
            raise RuntimeError("video boom")


def test_pipeline_runner_routes_through_services_for_group_and_video_flow():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": True,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    config_service = FakeConfigService(config)
    directory_service = FakeDirectoryService()
    masking_service = FakeMaskingService([])
    group_service = FakeGroupProcessingService()
    video_service = FakeVideoService()

    result = PipelineRunner(
        config_service=config_service,
        directory_service=directory_service,
        masking_service=masking_service,
        group_processing_service=group_service,
        video_service=video_service,
    ).run("config.json", overrides={"fps": 9})

    assert config_service.calls == [("config.json", {"fps": 9})]
    assert directory_service.prepared == [config]
    assert directory_service.cleared == ["frames"]
    assert len(group_service.calls) == 1
    assert len(video_service.calls) == 1
    image_folder, output_video_path, fps, kwargs = video_service.calls[0]
    assert (image_folder, output_video_path, fps) == ("frames", "videos/projection.mp4", 6)
    assert kwargs["log_progress"] is False
    assert kwargs["reporter"] is not None
    assert result is not None
    assert result.status == "completed"
    assert result.job_run.groups_total == 1
    assert result.job_run.groups_completed == 1
    assert {artifact.kind for artifact in result.job_run.artifacts} == {"output_dir", "projection_video"}
    assert result.output_image_folder == "frames"
    assert result.output_video_folder == "videos"
    assert result.output_json_folder == "json"
    assert result.projection_video_path == "videos/projection.mp4"


def test_pipeline_runner_uses_masked_groups_and_cleans_frames_when_configured():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": True,
        "clean_frames_after_video": True,
        "mask_videos": True,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "original.mp4", "camera_nr": 1}]],
    }
    masked_groups = [[{"path": "masked.mp4", "camera_nr": 1}]]
    directory_service = FakeDirectoryService()
    masking_service = FakeMaskingService(masked_groups)
    group_service = FakeGroupProcessingService()
    video_service = FakeVideoService()

    result = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=directory_service,
        masking_service=masking_service,
        group_processing_service=group_service,
        video_service=video_service,
    ).run("config.json")

    assert len(masking_service.calls) == 1
    masked_config, kwargs = masking_service.calls[0]
    assert masked_config == config
    assert kwargs["log_progress"] is False
    assert kwargs["reporter"] is not None
    assert group_service.calls[0][0][1] == masked_groups[0]
    assert directory_service.cleared == ["frames", "frames"]
    assert result is not None
    assert result.status == "completed"


def test_pipeline_runner_can_publish_events_to_external_observer():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    observer = InMemoryJobStore()

    result = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-123")

    mirrored = observer.get("job-123")

    assert result is not None
    assert mirrored is not None
    assert mirrored.status == "completed"
    assert [event.event_type for event in mirrored.events] == [
        "job_started",
        "config_loaded",
        "output_dirs_prepared",
        "artifact_created",
        "artifact_created",
        "artifact_created",
        "output_frames_cleared",
        "groups_discovered",
        "group_completed",
        "job_completed",
    ]


def test_pipeline_runner_marks_job_cancelled_before_group_processing():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": True,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    cancellation_token = CancellationToken()
    cancellation_token.cancel()

    group_service = FakeGroupProcessingService()
    video_service = FakeVideoService()
    result = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=group_service,
        video_service=video_service,
    ).run("config.json", cancellation_token=cancellation_token)

    assert result is not None
    assert result.status == "cancelled"
    assert result.job_run.events[-1].payload["error_code"] == "job_cancelled"
    assert group_service.calls == []
    assert video_service.calls == []


def test_pipeline_runner_can_run_request_from_in_memory_config():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    config_service = FakeConfigService(config)

    result = PipelineRunner(
        config_service=config_service,
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run_request(
        RunRequest(config=PipelineConfig.from_mapping(config), overrides={"fps": 9})
    )

    assert config_service.calls == []
    assert len(config_service.normalize_calls) == 1
    normalized_config, overrides = config_service.normalize_calls[0]
    assert isinstance(normalized_config, PipelineConfig)
    assert overrides == {"fps": 9}
    assert result is not None
    assert result.job_run.config_path == "<in-memory>"
    assert result.status == "completed"


def test_run_request_requires_exactly_one_config_source():
    with pytest.raises(ValueError):
        RunRequest()

    with pytest.raises(ValueError):
        RunRequest(config_path="config.json", config={})


def test_pipeline_runner_emits_error_code_for_config_failure():
    observer = InMemoryJobStore()

    result = PipelineRunner(
        config_service=FakeConfigService({}),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-config-fail")

    assert result is not None
    payload = observer.get("job-config-fail").events[-1].payload
    assert payload["error_code"] == "config_load_failed"


def test_pipeline_runner_captures_config_loader_error_details():
    class RaisingConfigService(FakeConfigService):
        def load(self, config_path, overrides=None):
            raise ValueError("bad config")

    observer = InMemoryJobStore()

    result = PipelineRunner(
        config_service=RaisingConfigService({}),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-config-error")

    assert result is not None
    event = observer.get("job-config-error").events[-1]
    assert event.event_type == "job_failed"
    assert event.payload["error_code"] == "config_load_failed"
    assert event.payload["error_detail"] == "bad config"


def test_pipeline_runner_emits_error_code_for_masking_failure():
    class FailingMaskingService(FakeMaskingService):
        def preprocess(self, config, **kwargs):
            raise RuntimeError("mask boom")

    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": True,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    observer = InMemoryJobStore()

    PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FailingMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-mask-fail")

    snapshot = observer.get("job-mask-fail")
    assert snapshot is not None
    masking_failed = [event for event in snapshot.events if event.event_type == "masking_failed"][0]
    assert masking_failed.payload["error_code"] == "masking_failed"
    assert masking_failed.payload["error_detail"] == "mask boom"


def test_pipeline_runner_emits_error_code_for_video_failure():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": True,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    observer = InMemoryJobStore()
    video_service = FakeVideoService()
    video_service.should_fail = True

    PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=video_service,
    ).run("config.json", observer=observer, job_id="job-video-fail")

    snapshot = observer.get("job-video-fail")
    assert snapshot is not None
    video_failed = [event for event in snapshot.events if event.event_type == "video_failed"][0]
    assert video_failed.payload["error_code"] == "video_failed"
    assert video_failed.payload["error_detail"] == "video boom"


def test_pipeline_runner_run_config_routes_through_request_normalization():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    config_service = FakeConfigService(config)

    result = PipelineRunner(
        config_service=config_service,
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run_config(config, overrides={"fps": 8})

    assert result is not None
    assert len(config_service.normalize_calls) == 1
    assert config_service.normalize_calls[0][1] == {"fps": 8}


def test_pipeline_runner_emits_groups_empty_when_no_groups():
    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [],
    }
    observer = InMemoryJobStore()

    result = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-empty")

    snapshot = observer.get("job-empty")
    assert result is not None
    assert snapshot is not None
    assert any(event.event_type == "groups_empty" for event in snapshot.events)


def test_pipeline_runner_handles_group_failure_and_marks_job_completed_with_errors():
    class FailingGroupProcessingService(FakeGroupProcessingService):
        def process_group(self, *args, **kwargs):
            raise RuntimeError("group boom")

    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    observer = InMemoryJobStore()

    result = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FailingGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-group-fail")

    snapshot = observer.get("job-group-fail")
    assert result is not None
    assert snapshot is not None
    group_failed = [event for event in snapshot.events if event.event_type == "group_failed"][0]
    assert group_failed.payload["error_code"] == "group_failed"
    assert snapshot.status == "completed"
    assert snapshot.events[-1].payload["had_errors"] is True


def test_pipeline_runner_handles_group_cancellation_via_token():
    class CancellingGroupProcessingService(FakeGroupProcessingService):
        def process_group(self, *args, reporter=None, cancellation_token=None):
            cancellation_token.cancel()
            raise JobCancelledError("cancelled")

    config = {
        "model_path": "models/yolo.pt",
        "fps": 6,
        "create_projection_video": False,
        "clean_frames_after_video": False,
        "mask_videos": False,
        "output_video_filename": "projection.mp4",
        "video_groups": [[{"path": "input.json", "camera_nr": 1}]],
    }
    observer = InMemoryJobStore()
    cancellation_token = CancellationToken()

    result = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=CancellingGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-group-cancel", cancellation_token=cancellation_token)

    snapshot = observer.get("job-group-cancel")
    assert result is not None
    assert snapshot is not None
    assert any(event.event_type == "group_cancelled" for event in snapshot.events)
    assert any(event.event_type == "job_cancelled" for event in snapshot.events)
    assert result.status == "cancelled"
