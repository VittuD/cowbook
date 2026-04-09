# Calibration

Cowbook currently uses a bundle-based calibration file:

- `assets/calibration/camera_system.json`

The bundle holds the geometry the engine needs at run time: world dimensions, default intrinsics, per-camera overrides, and the correspondence points used to project image detections into barn coordinates. Today the package supports both `pinhole` and `fisheye` camera models.

The calibration subsystem is part of the engine layer because [projection](pipeline.md) is part of the run itself. The engine executes against explicit runtime calibration artifacts.

`assets/calibration/camera_correspondences.json` is still available as an auxiliary correspondence source, but the canonical runtime calibration asset is the bundle file above.

Cowbook's calibration math is built on OpenCV's calibration and projection stack. See [References and Attribution](references.md) for the upstream documentation links.
