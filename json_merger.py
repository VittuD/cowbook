# json_merger.py
"""
Merge multiple tracking JSONs into one.
- Works best with *_processed.json produced by frame processing (has centroids & projected_centroids).
- Frames are aligned by frame_id; merged output includes all frame_ids present (up to max).
- For each frame, detections from all inputs are concatenated.
- Label IDs are reassigned sequentially per frame starting at 1.
- class_id is preserved when available.

CLI:
    python json_merger.py --inputs a_processed.json b_processed.json ... --output merged_processed.json
"""

import json
import argparse
from typing import List, Dict, Any


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _compute_centroids_from_xyxy(xyxy_list: List[List[float]]) -> List[List[float]]:
    # Fallback if centroids missing (we prefer processed inputs, but this keeps it robust)
    out = []
    for x1, y1, x2, y2 in xyxy_list:
        out.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    return out


def _reassign_labels_sequential(total_dets: int, merged_labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build labels with id=1..total_dets; preserve class_id when available.
    """
    out = []
    for i in range(total_dets):
        class_id = None
        if i < len(merged_labels) and isinstance(merged_labels[i], dict):
            class_id = merged_labels[i].get("class_id")
        out.append({"class_id": class_id, "id": i + 1})
    return out


def merge_json_files(input_files: List[str], output_file: str) -> None:
    """
    Merge JSONs into one, including centroids & projected_centroids if present.

    - Non-matching numbers of frames are OK: we include all frame_ids seen.
    - For each frame_id, we concatenate all detections and labels (then reassign ids).
    - For fields under detections, we include:
        - always: xyxy
        - when present in any input: centroids, projected_centroids
      If a file is missing 'centroids' but has 'xyxy', we compute centroids from boxes as fallback.
      For 'projected_centroids' we include only when provided (no recomputation here).
    """
    frames_acc: Dict[int, Dict[str, Any]] = {}

    for path in input_files:
        data = _load_json(path)
        for frame in data.get("frames", []):
            fid = int(frame.get("frame_id", 0))
            dets = frame.get("detections", {}) or {}
            xyxy = dets.get("xyxy", []) or []
            cents = dets.get("centroids", None)  # might be None
            projs = dets.get("projected_centroids", None)  # might be None
            labels = frame.get("labels", []) or []

            if fid not in frames_acc:
                frames_acc[fid] = {
                    "xyxy": [],
                    "centroids": None,            # lazily allocated if ever present/needed
                    "projected_centroids": None,  # lazily allocated if ever present
                    "labels": [],
                }

            bucket = frames_acc[fid]
            # xyxy always extended
            bucket["xyxy"].extend(xyxy)

            # centroids: include if present in this file; else fallback compute from xyxy for this segment
            if cents is not None:
                if bucket["centroids"] is None:
                    bucket["centroids"] = []
                bucket["centroids"].extend(cents)
            else:
                # if we already decided to include centroids (some file had them), keep lengths in sync
                if bucket["centroids"] is not None and xyxy:
                    bucket["centroids"].extend(_compute_centroids_from_xyxy(xyxy))

            # projected_centroids: only extend if present
            if projs is not None:
                if bucket["projected_centroids"] is None:
                    bucket["projected_centroids"] = []
                bucket["projected_centroids"].extend(projs)
            # If absent, do nothing (we do not invent projected coordinates)

            # labels (will be re-id'd later)
            # Ensure labels length matches xyxy length from this source segment
            if len(labels) < len(xyxy):
                labels = labels + [{"class_id": None, "id": None} for _ in range(len(xyxy) - len(labels))]
            elif len(labels) > len(xyxy):
                labels = labels[: len(xyxy)]
            bucket["labels"].extend(labels)

    # Build merged frames ordered by frame_id
    merged_frames: List[Dict[str, Any]] = []
    for fid in sorted(frames_acc.keys()):
        bucket = frames_acc[fid]
        xyxy = bucket["xyxy"]
        dets_out = {"xyxy": xyxy}

        # include centroids if we ever collected them for this fid
        if bucket["centroids"] is not None:
            dets_out["centroids"] = bucket["centroids"]
        # include projected_centroids only if present
        if bucket["projected_centroids"] is not None:
            dets_out["projected_centroids"] = bucket["projected_centroids"]

        labels = _reassign_labels_sequential(len(xyxy), bucket["labels"])
        merged_frames.append({"frame_id": fid, "detections": dets_out, "labels": labels})

    with open(output_file, "w") as f:
        json.dump({"frames": merged_frames}, f, indent=4)

    print(
        f"Merged {len(input_files)} JSONs into {output_file} "
        f"({len(merged_frames)} frames; max frame_id={max(frames_acc) if frames_acc else 'N/A'})"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge tracking JSON files (processed or raw) into one")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSON files to merge")
    p.add_argument("--output", required=True, help="Output merged JSON path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge_json_files(args.inputs, args.output)
