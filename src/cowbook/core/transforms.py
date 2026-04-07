from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from cowbook.core.contracts import Detections, TrackingDocument, TrackingFrame, TrackingLabel


def centroid_from_xyxy(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_wh_area(box: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w, h, w * h)


def normalize_labels_len(labels: list[dict[str, Any]] | None, n: int) -> list[dict[str, Any]]:
    labels = labels or []
    if len(labels) < n:
        labels = labels + [{"class_id": None, "id": None} for _ in range(n - len(labels))]
    elif len(labels) > n:
        labels = labels[:n]
    return labels


def convert_arrays_to_lists(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: convert_arrays_to_lists(value) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_arrays_to_lists(element) for element in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def extract_frames_data(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    frames_data = []
    for frame in json_data["frames"]:
        frame_data = {
            "frame_id": frame["frame_id"],
            "detections": [
                {
                    "bbox": bbox,
                    "centroid": list(centroid_from_xyxy(bbox)),
                }
                for bbox in frame["detections"]["xyxy"]
            ],
            "labels": [
                TrackingLabel.from_mapping(label).to_dict()
                for label in frame.get("labels", [])
            ],
        }
        frames_data.append(frame_data)
    return frames_data


def reconstruct_tracking_document(frames_data: list[dict[str, Any]]) -> dict[str, Any]:
    frames: list[TrackingFrame] = []
    for frame_data in frames_data:
        detections = frame_data["detections"]
        frames.append(
            TrackingFrame(
                frame_id=frame_data["frame_id"],
                detections=Detections(
                    xyxy=[convert_arrays_to_lists(detection["bbox"]) for detection in detections],
                    centroids=[convert_arrays_to_lists(detection["centroid"]) for detection in detections],
                    projected_centroids=[
                        convert_arrays_to_lists(detection.get("projected_centroid"))
                        for detection in detections
                    ],
                ),
                labels=[
                    TrackingLabel.from_mapping(label)
                    for label in frame_data.get("labels", [])
                ],
            )
        )
    return TrackingDocument(frames=frames).to_dict()


def aggregate_projected_centroids(documents: Iterable[dict[str, Any]]) -> dict[int, list[Any]]:
    all_projected_centroids: dict[int, list[Any]] = {}
    for data in documents:
        for frame in data.get("frames", []):
            frame_id = frame["frame_id"]
            projected_centroids = frame["detections"].get("projected_centroids", [])
            if frame_id not in all_projected_centroids:
                all_projected_centroids[frame_id] = []
            all_projected_centroids[frame_id].extend(projected_centroids)
    return all_projected_centroids


def merge_tracking_documents(
    documents: Iterable[dict[str, Any]],
    *,
    camera_nrs: Iterable[int | None] | None = None,
) -> dict[str, Any]:
    frames_acc: dict[int, dict[str, Any]] = {}

    document_list = list(documents)
    camera_nr_list = list(camera_nrs) if camera_nrs is not None else [None] * len(document_list)
    if len(camera_nr_list) != len(document_list):
        raise ValueError("camera_nrs must align 1:1 with the input documents")

    for data, source_camera_nr in zip(document_list, camera_nr_list):
        for frame in data.get("frames", []):
            fid = int(frame.get("frame_id", 0))
            dets = frame.get("detections", {}) or {}
            xyxy = dets.get("xyxy", []) or []
            cents = dets.get("centroids", None)
            projs = dets.get("projected_centroids", None)
            labels = frame.get("labels", []) or []

            if fid not in frames_acc:
                frames_acc[fid] = {
                    "xyxy": [],
                    "centroids": None,
                    "projected_centroids": None,
                    "labels": [],
                }

            bucket = frames_acc[fid]
            bucket["xyxy"].extend(xyxy)

            if cents is not None:
                if bucket["centroids"] is None:
                    bucket["centroids"] = []
                bucket["centroids"].extend(cents)
            elif bucket["centroids"] is not None and xyxy:
                bucket["centroids"].extend([list(centroid_from_xyxy(box)) for box in xyxy])

            if projs is not None:
                if bucket["projected_centroids"] is None:
                    bucket["projected_centroids"] = []
                bucket["projected_centroids"].extend(projs)

            normalized_labels = normalize_labels_len(labels, len(xyxy))
            for label in normalized_labels:
                tracking_label = TrackingLabel.from_mapping(label)
                bucket["labels"].append(
                    TrackingLabel(
                        class_id=tracking_label.class_id,
                        id=None,
                        camera_nr=tracking_label.camera_nr or source_camera_nr,
                        local_track_id=tracking_label.local_track_id or tracking_label.id,
                        global_id=tracking_label.global_id,
                    )
                )

    merged_frames: list[TrackingFrame] = []
    for fid in sorted(frames_acc.keys()):
        bucket = frames_acc[fid]
        xyxy = bucket["xyxy"]
        labels = [
            bucket["labels"][i]
            for i in range(len(xyxy))
        ]
        merged_frames.append(
            TrackingFrame(
                frame_id=fid,
                detections=Detections(
                    xyxy=xyxy,
                    centroids=bucket["centroids"],
                    projected_centroids=bucket["projected_centroids"],
                ),
                labels=labels,
            )
        )

    return TrackingDocument(frames=merged_frames).to_dict()


def iter_csv_rows(
    document: dict[str, Any],
    *,
    source_tag: str | None = None,
    source_key: str = "source",
) -> Iterable[dict[str, Any]]:
    for frame in document.get("frames", []) or []:
        fid = int(frame.get("frame_id", 0))
        dets = frame.get("detections", {}) or {}
        xyxy = dets.get("xyxy", []) or []
        cents = dets.get("centroids", None)
        projs = dets.get("projected_centroids", None)

        labels = normalize_labels_len(frame.get("labels", []) or [], len(xyxy))

        for i, box in enumerate(xyxy):
            lab = labels[i] if 0 <= i < len(labels) else {}
            class_id = lab.get("class_id")
            det_id = lab.get("global_id")
            if det_id is None:
                det_id = lab.get("local_track_id", lab.get("id"))

            c = cents[i] if cents and 0 <= i < len(cents) else None
            if c is not None and isinstance(c, (list, tuple)) and len(c) >= 2:
                cx, cy = float(c[0]), float(c[1])
            else:
                cx, cy = centroid_from_xyxy(box)

            proj_x = proj_y = proj_z = None
            p = projs[i] if projs and 0 <= i < len(projs) else None
            if p is not None and isinstance(p, (list, tuple)) and len(p) >= 2:
                proj_x, proj_y = float(p[0]), float(p[1])
                if len(p) >= 3:
                    proj_z = float(p[2])

            w, h, area = bbox_wh_area(box)
            row = {
                "frame_id": fid,
                "det_index": i + 1,
                "id": det_id,
                "class_id": class_id,
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "w": w,
                "h": h,
                "area": area,
                "centroid_x": cx,
                "centroid_y": cy,
                "proj_x": proj_x,
                "proj_y": proj_y,
                "proj_z": proj_z,
            }
            if source_tag is not None:
                row[source_key] = source_tag
            yield row
