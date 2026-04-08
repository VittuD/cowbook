from __future__ import annotations

from cowbook.core.contracts import PipelineConfig
from cowbook.execution.models import JobArtifact, JobRun
from cowbook.execution.results import build_run_result


def test_build_run_result_summarizes_artifacts_by_kind():
    job_run = JobRun(
        job_id="job-1",
        config_path="config.json",
        artifacts=[
            JobArtifact(kind="output_dir", path="frames", metadata={"role": "frames"}),
            JobArtifact(kind="output_dir", path="videos", metadata={"role": "videos"}),
            JobArtifact(kind="output_dir", path="json", metadata={"role": "json"}),
            JobArtifact(kind="tracking_json", path="json/cam1_tracking.json"),
            JobArtifact(kind="tracking_json", path="json/cam1_tracking.json"),
            JobArtifact(kind="processed_json", path="json/cam1_tracking_processed.json"),
            JobArtifact(kind="merged_json", path="json/group_1_merged_processed.json"),
            JobArtifact(kind="csv", path="json/group_1_merged_processed.csv"),
            JobArtifact(kind="projection_video", path="videos/projection.mp4"),
        ],
    )

    result = build_run_result(
        job_run,
        resolved_config=PipelineConfig(
            output_root="var/runs/demo",
            output_image_folder="var/runs/demo/frames",
            output_video_folder="var/runs/demo/videos",
            output_json_folder="var/runs/demo/json",
        ),
    )

    assert result.output_root == "var/runs/demo"
    assert result.output_image_folder == "frames"
    assert result.output_video_folder == "videos"
    assert result.output_json_folder == "json"
    assert result.projection_video_path == "videos/projection.mp4"
    assert result.tracking_json_paths == ["json/cam1_tracking.json"]
    assert result.processed_json_paths == ["json/cam1_tracking_processed.json"]
    assert result.merged_json_paths == ["json/group_1_merged_processed.json"]
    assert result.csv_paths == ["json/group_1_merged_processed.csv"]
