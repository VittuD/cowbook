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

import argparse
import json
import logging
from typing import Any, Dict, List

from cowbook.contracts import TrackingDocument
from cowbook.transforms import centroid_from_xyxy, merge_tracking_documents

logger = logging.getLogger(__name__)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return TrackingDocument.from_mapping(json.load(f)).to_dict()


def _compute_centroids_from_xyxy(xyxy_list: List[List[float]]) -> List[List[float]]:
    # Fallback if centroids missing (we prefer processed inputs, but this keeps it robust)
    return [list(centroid_from_xyxy(box)) for box in xyxy_list]


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
    documents = [_load_json(path) for path in input_files]
    merged_doc = merge_tracking_documents(documents)

    with open(output_file, "w") as f:
        json.dump(merged_doc, f, indent=4)

    logger.info(
        f"Merged {len(input_files)} JSONs into {output_file} "
        f"({len(merged_doc['frames'])} frames; "
        f"max frame_id={max((frame['frame_id'] for frame in merged_doc['frames']), default='N/A')})"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge tracking JSON files (processed or raw) into one")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSON files to merge")
    p.add_argument("--output", required=True, help="Output merged JSON path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge_json_files(args.inputs, args.output)
