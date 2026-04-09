from __future__ import annotations

from cowbook.core.contracts import (
    Detections,
    PipelineConfig,
    TrackingCleanupConfig,
    TrackingDocument,
    TrackingFrame,
    TrackingLabel,
    VideoGroupItem,
)


def test_pipeline_config_round_trip_matches_current_shape():
    config = PipelineConfig.from_mapping(
        {
            "model_path": "models/custom.pt",
            "fps": "12",
            "output_image_format": "jpg",
            "video_groups": [[{"path": "videos/a.mp4", "camera_nr": "1"}]],
            "camera_to_mask_map": {1: "Ch1"},
        }
    )

    data = config.to_dict()

    assert data["model_path"] == "models/custom.pt"
    assert data["fps"] == 12
    assert data["video_groups"] == [[{"path": "videos/a.mp4", "camera_nr": 1}]]
    assert data["camera_to_mask_map"] == {"1": "Ch1"}
    assert data["runtime_root"] == "var"
    assert data["run_name"] == "default"
    assert data["output_root"] == "var/runs/default"
    assert data["masks"]["Ch1"] == "assets/masks/combined_mask_ch1.png"
    assert data["tracking_cleanup"]["enabled"] is False


def test_tracking_document_round_trip_for_raw_shape(raw_tracking_doc):
    document = TrackingDocument.from_mapping(raw_tracking_doc)

    assert document.to_dict() == raw_tracking_doc


def test_tracking_document_round_trip_for_processed_shape(processed_tracking_doc):
    document = TrackingDocument.from_mapping(processed_tracking_doc)

    assert document.to_dict() == processed_tracking_doc


def test_tracking_frame_serialization_preserves_optional_fields():
    frame = TrackingFrame(
        frame_id=5,
        detections=Detections(
            xyxy=[[1.0, 2.0, 3.0, 4.0]],
            centroids=[[2.0, 3.0]],
            projected_centroids=[[100.0, 200.0, 100.0]],
        ),
        labels=[TrackingLabel(class_id=0, id=1)],
    )

    assert frame.to_dict() == {
        "frame_id": 5,
        "detections": {
            "xyxy": [[1.0, 2.0, 3.0, 4.0]],
            "centroids": [[2.0, 3.0]],
            "projected_centroids": [[100.0, 200.0, 100.0]],
        },
        "labels": [{"class_id": 0, "id": 1}],
    }


def test_tracking_label_round_trip_preserves_cleanup_metadata():
    label = TrackingLabel.from_mapping(
        {
            "class_id": 0,
            "id": 7,
            "det_idx": 3,
            "real": 0,
            "src": "gap_fill",
        }
    )

    assert label.to_dict() == {
        "class_id": 0,
        "id": 7,
        "det_idx": 3,
        "real": 0,
        "src": "gap_fill",
    }


def test_tracking_cleanup_config_round_trip_preserves_nested_shape():
    cleanup = TrackingCleanupConfig.from_mapping(
        {
            "enabled": True,
            "roi": [[1, 2], [3, 4], [5, 6]],
            "two_pass_prune_short_tracks": True,
            "min_track_total_observations": 12,
            "postprocess_gap_fill": True,
        }
    )

    data = cleanup.to_dict()

    assert data["enabled"] is True
    assert data["roi"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert data["two_pass_prune_short_tracks"] is True
    assert data["min_track_total_observations"] == 12
    assert data["postprocess_gap_fill"] is True


def test_video_group_item_normalizes_camera_number():
    item = VideoGroupItem.from_mapping({"path": "videos/a.mp4", "camera_nr": "4"})

    assert item.to_dict() == {"path": "videos/a.mp4", "camera_nr": 4}
