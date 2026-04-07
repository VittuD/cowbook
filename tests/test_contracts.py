from __future__ import annotations

from cowbook.core.contracts import (
    Detections,
    PipelineConfig,
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
    assert data["masks"]["Ch1"] == "test_img/combined_mask_ch1.png"


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


def test_video_group_item_normalizes_camera_number():
    item = VideoGroupItem.from_mapping({"path": "videos/a.mp4", "camera_nr": "4"})

    assert item.to_dict() == {"path": "videos/a.mp4", "camera_nr": 4}
