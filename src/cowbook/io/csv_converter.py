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

import argparse
import csv
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cowbook.core.contracts import TrackingDocument
from cowbook.core.transforms import (
    bbox_wh_area,
    centroid_from_xyxy,
    iter_csv_rows,
    normalize_labels_len,
)

logger = logging.getLogger(__name__)


# --------- helpers ---------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return TrackingDocument.from_mapping(json.load(f)).to_dict()


def _centroid_from_xyxy(b: List[float]) -> Tuple[float, float]:
    return centroid_from_xyxy(b)


def _bbox_wh_area(b: List[float]) -> Tuple[float, float, float]:
    return bbox_wh_area(b)


def _safe_get(lst: Optional[List[Any]], idx: int) -> Optional[Any]:
    if not lst:
        return None
    if 0 <= idx < len(lst):
        return lst[idx]
    return None


def _normalize_labels_len(labels: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pad/truncate labels to length n."""
    return normalize_labels_len(labels, n)


# --------- core row extraction ---------

def _iter_rows_from_json(doc: Dict[str, Any], source_tag: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """Yield one CSV row per detection per frame."""
    yield from iter_csv_rows(doc, source_tag=source_tag)


def _fieldnames(
    source_col: Optional[str] = None,
    *,
    include_source: Optional[bool] = None,
) -> List[str]:
    if include_source is not None:
        source_col = "source" if include_source else None
    core = [
        "frame_id", "det_index", "id", "class_id",
        "x1", "y1", "x2", "y2", "w", "h", "area",
        "centroid_x", "centroid_y",
        "proj_x", "proj_y", "proj_z",
    ]
    return ([source_col] + core) if source_col else core


def _write_csv(
    rows: Iterable[Dict[str, Any]],
    out_path: str,
    source_col: Optional[str] = None,
    *,
    include_source: Optional[bool] = None,
) -> None:
    fn = _fieldnames(source_col, include_source=include_source)
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
    def _rows():
        for path in inputs:
            doc = _load_json(path)
            for row in _iter_rows_from_json(doc, source_tag=None):
                if args.source_col is not None:
                    row[args.source_col] = os.path.basename(path)
                yield row

    _write_csv(_rows(), out_path, source_col=args.source_col)
    logger.info("Wrote CSV: %s (from %d JSON file(s))", out_path, len(inputs))


if __name__ == "__main__":
    main()
