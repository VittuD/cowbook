# References and Attribution

Cowbook builds on a small set of external libraries and standard computer-vision workflows. This page records the important ones so the package docs stay clear about what is project code and what comes from upstream tooling.

## Detection and Tracking

Object detection and tracking are currently run through [Ultralytics YOLO](https://docs.ultralytics.com/), using the tracking mode documented at [docs.ultralytics.com/modes/track](https://docs.ultralytics.com/modes/track/). Cowbook wraps that runtime behavior and converts the results into its own JSON and execution contracts.

The project-specific model weights used with this repository are not part of the `cowbook` engine package. They are project artifacts, intended for non-commercial use, and published separately by the project team.

## Geometry and Calibration

Camera geometry, projection, and intrinsic handling rely on OpenCV. The underlying camera-model and projection machinery comes from OpenCV's `calib3d` module, documented at [docs.opencv.org/4.x/d9/d0c/group__calib3d.html](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). Future intrinsic workflows based on ChArUco boards will use OpenCV's ChArUco support, documented at [docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html).

## Image and Video Processing

Masking, frame-level preprocessing, and image IO are also built on OpenCV.

## Documentation Tooling

The package documentation is built with [MkDocs](https://www.mkdocs.org/), [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), and [mkdocstrings](https://mkdocstrings.github.io/).
