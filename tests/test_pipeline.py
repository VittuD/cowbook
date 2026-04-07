from __future__ import annotations

from cowbook.app.pipeline import PipelineRunner
from cowbook.execution import CancellationToken, InMemoryJobStore


class FakeConfigService:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def load(self, config_path, overrides=None):
        self.calls.append((config_path, overrides))
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

    def preprocess(self, config):
        self.calls.append(config)
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

    def create_projection_video(self, image_folder, output_video_path, fps):
        self.calls.append((image_folder, output_video_path, fps))


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

    snapshot = PipelineRunner(
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
    assert video_service.calls == [("frames", "videos/projection.mp4", 6)]
    assert snapshot.status == "completed"
    assert snapshot.groups_total == 1
    assert snapshot.groups_completed == 1
    assert {artifact.kind for artifact in snapshot.artifacts} == {"output_dir", "projection_video"}


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

    snapshot = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=directory_service,
        masking_service=masking_service,
        group_processing_service=group_service,
        video_service=video_service,
    ).run("config.json")

    assert masking_service.calls == [config]
    assert group_service.calls[0][0][1] == masked_groups[0]
    assert directory_service.cleared == ["frames", "frames"]
    assert snapshot.status == "completed"


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

    snapshot = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=FakeGroupProcessingService(),
        video_service=FakeVideoService(),
    ).run("config.json", observer=observer, job_id="job-123")

    mirrored = observer.get("job-123")

    assert snapshot is not None
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
    snapshot = PipelineRunner(
        config_service=FakeConfigService(config),
        directory_service=FakeDirectoryService(),
        masking_service=FakeMaskingService([]),
        group_processing_service=group_service,
        video_service=video_service,
    ).run("config.json", cancellation_token=cancellation_token)

    assert snapshot is not None
    assert snapshot.status == "cancelled"
    assert group_service.calls == []
    assert video_service.calls == []
