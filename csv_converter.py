# csv_converter.py
"""
Convert tracking JSON (raw or processed) to a tabular CSV.

Supported inputs
----------------
- Raw tracking JSON from tracking.py:
    frames[].detections.xyxy, frames[].labels[{class_id, id}]
- Processed JSON from frame_processor/reconstruct_json:
    frames[].detections.xyxy
    frames[].detections.centroids
    frames[].detections.projected_centroids
    frames[].labels[{class_id, id}]

Behavior
--------
- One output row per detection per frame.
- If centroids are missing, they are computed from xyxy.
- If projected_centroids are missing, columns are left empty.
- If labels length does not match detections, it is padded/truncated safely.
- Can accept multiple input files and write one combined CSV.

CLI
---
Examples:
    # Single file
    python csv_converter.py --input output_json/group_1_merged_processed.json --output group1.csv

    # Multiple files -> one CSV, with a 'source' column indicating which file each row came from
    python csv_converter.py --inputs output_json/group_1_merged_processed.json output_json/group_2_merged_processed.json \
                            --output all_groups.csv --source-col source
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------- helpers ---------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _centroid_from_xyxy(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_wh_area(b: List[float]) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w, h, w * h)


def _safe_get(lst: Optional[List[Any]], idx: int) -> Optional[Any]:
    if not lst:
        return None
    if 0 <= idx < len(lst):
        return lst[idx]
    return None


def _normalize_labels_len(labels: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pad/truncate labels to length n."""
    if labels is None:
        labels = []
    if len(labels) < n:
        labels = labels + [{"class_id": None, "id": None} for _ in range(n - len(labels))]
    elif len(labels) > n:
        labels = labels[:n]
    return labels


# --------- core row extraction ---------

def _iter_rows_from_json(doc: Dict[str, Any], source_tag: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """Yield one CSV row per detection per frame."""
    frames: List[Dict[str, Any]] = doc.get("frames", []) or []

    for frame in frames:
        fid = int(frame.get("frame_id", 0))
        dets = frame.get("detections", {}) or {}
        xyxy = dets.get("xyxy", []) or []
        cents = dets.get("centroids", None)  # may be None
        projs = dets.get("projected_centroids", None)  # may be None (2D or 3D)

        labels = frame.get("labels", []) or []
        labels = _normalize_labels_len(labels, len(xyxy))

        for i, box in enumerate(xyxy):
            # label fields
            lab = _safe_get(labels, i) or {}
            class_id = lab.get("class_id")
            det_id = lab.get("id")

            # centroid (from JSON or computed)
            cx, cy = (None, None)
            c = _safe_get(cents, i)
            if c is not None and isinstance(c, (list, tuple)) and len(c) >= 2:
                cx, cy = float(c[0]), float(c[1])
            else:
                cx, cy = _centroid_from_xyxy(box)

            # projected centroid (if present)
            proj_x = proj_y = proj_z = None
            p = _safe_get(projs, i)
            if p is not None and isinstance(p, (list, tuple)) and len(p) >= 2:
                proj_x, proj_y = float(p[0]), float(p[1])
                if len(p) >= 3:
                    proj_z = float(p[2])

            # bbox derived
            w, h, area = _bbox_wh_area(box)

            row = {
                "frame_id": fid,
                "det_index": i + 1,   # 1-based index within the frame
                "id": det_id,         # per-frame id (already reassigned elsewhere)
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
                row["source"] = source_tag
            yield row


def _fieldnames(include_source: bool) -> List[str]:
    core = [
        "frame_id", "det_index", "id", "class_id",
        "x1", "y1", "x2", "y2", "w", "h", "area",
        "centroid_x", "centroid_y",
        "proj_x", "proj_y", "proj_z",
    ]
    return (["source"] + core) if include_source else core


def _write_csv(rows: Iterable[Dict[str, Any]], out_path: str, include_source: bool) -> None:
    fn = _fieldnames(include_source)
    # Ensure folder exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --------- CLI ---------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert tracking JSON (raw/processed) to CSV.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Single input JSON file")
    g.add_argument("--inputs", nargs="+", help="Multiple input JSON files")

    p.add_argument("--output", required=False, help="Output CSV path. "
                                                    "Default: <input>.csv for single input; 'merged.csv' for multiple.")
    p.add_argument("--source-col", default=None, help="Optional column name to include the source file label (e.g., 'source').")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    inputs: List[str]
    if args.input:
        inputs = [args.input]
        default_out = os.path.splitext(args.input)[0] + ".csv"
    else:
        inputs = list(args.inputs)
        default_out = "merged.csv"

    out_path = args.output or default_out
    include_source = args.source_col is not None

    def _rows():
        for path in inputs:
            doc = _load_json(path)
            source_tag = None
            if include_source:
                # Use the provided column name, but the value is the basename of the file
                # We'll attach the key in _write_csv via extrasaction="ignore" pattern,
                # so add 'source' key to each row with the exact column name requested.
                # Simpler: inject with a consistent key then remap if the user wants a different header.
                pass
            # We want the column header to match args.source_col; easiest is to yield rows with 'source',
            # then post-rename the header if user asked for a custom name. Instead, simpler:
            # put the desired key directly into rows.
            for row in _iter_rows_from_json(doc, source_tag=None):
                if include_source:
                    row[args.source_col] = os.path.basename(path)
                yield row

    _write_csv(_rows(), out_path, include_source=True if include_source else False)
    print(f"Wrote CSV: {out_path} (from {len(inputs)} JSON file(s))")


if __name__ == "__main__":
    main()
