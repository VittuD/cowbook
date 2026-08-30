# Pipeline

Cowbook runs a fixed batch pipeline:

`video(s) or tracking json -> raw tracking json -> calibration-coordinate normalization -> processed json -> merged group json -> rendered frames -> mp4`

The important terms are:

- `source image coordinates`: raw distorted pixels in the input frame resolution
- `calibration image coordinates`: pixels in the resolution declared by the camera calibration bundle
- `centroid`: the point derived from each detection box after coordinate normalization and undistortion
- `projected centroid`: that centroid after camera undistortion and projection into barn coordinates
- `group-level merge`: the merged processed output for one configured camera group

The pipeline stages are:

1. load and normalize config
2. prepare run-scoped output directories
3. optionally preprocess videos with masks
4. run tracking for video inputs and emit raw tracking JSON with its source image size
5. map raw detection pixels into calibration image coordinates when source and calibration sizes differ
6. undistort detection coordinates and compute projected centroids for each camera JSON
7. render combined projected frames for the group
8. merge processed JSONs into one group-level document
9. optionally export CSVs
10. optionally assemble the final MP4

If one camera in a group fails, the group continues with the surviving cameras.

When `tracking_cleanup.enabled` is true, the tracking stage expands into:

1. detection cache for the current run
2. detection preprocessing
   this can apply confidence, ROI, edge, aspect, absolute-area, frame-area-ratio, and optional mask-fill filters before NMS
3. cleanup tracking pass 1
4. optional short-track pruning by gap-tolerant consecutive streak plus optional total-observation threshold
5. optional cleanup tracking pass 2
6. optional temporal postprocessing
   temporal postprocessing can run gap fill, smoothing, or both depending on config

`min_track_length` is evaluated against the longest surviving streak for each track, not against total lifetime observations. `short_track_gap_tolerance` controls how many missing frames are tolerated inside that streak; the default is `6`. `min_track_total_observations`, when set, adds a second requirement on overall observation count.

## Tracking Document Coordinates

A resolution-aware raw tracking document has this shape:

```json
{
  "source_image_size": [1280, 720],
  "frames": [
    {
      "frame_id": 0,
      "detections": {"xyxy": [[100.0, 200.0, 180.0, 300.0]]},
      "labels": []
    }
  ]
}
```

`detections.xyxy` is expressed in `source_image_size`. Tracking documents created from video inputs include this field automatically. Precomputed JSON should provide it whenever its detections are not already in the calibration image coordinate system.

Per-camera processed JSON may additionally contain:

```json
{
  "source_image_size": [1280, 720],
  "calibration_image_size": [2688, 1520],
  "coordinate_space": "undistorted_calibration_image"
}
```

Merged multi-camera results share barn-space `projected_centroids`; their image-space boxes do not define one common camera image coordinate system. Metadata-free legacy raw JSON remains accepted and is assumed to already use calibration image coordinates.
