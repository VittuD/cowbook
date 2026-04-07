# Pipeline

Cowbook runs a fixed batch pipeline:

`video(s) or tracking json -> raw tracking json -> processed json -> merged group json -> rendered frames -> mp4`

The important terms are:

- `centroid`: the point derived from each detection box after processing
- `projected centroid`: that centroid after camera undistortion and projection into barn coordinates
- `group-level merge`: the merged processed output for one configured camera group

The pipeline stages are:

1. load and normalize config
2. prepare run-scoped output directories
3. optionally preprocess videos with masks
4. run tracking for video inputs and emit raw tracking JSON
5. compute centroids and projected centroids for each camera JSON
6. render combined projected frames for the group
7. merge processed JSONs into one group-level document
8. optionally export CSVs
9. optionally assemble the final MP4

If one camera in a group fails, the group continues with the surviving cameras.
