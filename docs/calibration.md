# Calibration

Cowbook currently uses a bundle-based calibration file:

- `assets/calibration/camera_system.json`

The bundle holds the geometry the engine needs at run time: world dimensions, default intrinsics, per-camera overrides, and correspondence points used to project image detections into barn coordinates. Today the package supports both `pinhole` and `fisheye` camera models.

## Calibration Image Size

An intrinsic camera matrix is valid in the pixel coordinate system at which it was calibrated. The canonical bundle therefore includes `image_size` alongside `camera_matrix` and `dist_coeff`:

```json
{
  "defaults": {
    "model_type": "pinhole",
    "image_size": [2688, 1520],
    "camera_matrix": [[1102, 0, 1356], [0, 1102, 774], [0, 0, 1]],
    "dist_coeff": [[0.6894, 0.0285, -0.0030, -0.0005, -0.0008]]
  }
}
```

`image_size` is `[width, height]`. New structured calibration bundles must provide a positive size. The legacy matrix-only artifact, `assets/calibration/calibration_matrix.json`, remains readable for compatibility but has no declared source resolution; the engine interprets it at the historical `2688x1520` default. Use a structured bundle for calibrations produced at any other resolution.

## Coordinate Adaptation

Raw tracking detections are measured in the input video's source pixel coordinates. When a tracking document contains `source_image_size`, Cowbook maps its detection boxes into the calibration pixel coordinate system before lens undistortion and ground-plane projection:

```text
x_calibration = x_source * calibration_width / source_width
y_calibration = y_source * calibration_height / source_height
```

This supports an input that is a pure resize of the calibrated camera image. Distortion coefficients stay unchanged because resizing changes the pixel intrinsics, not the physical lens model.

For output video rectification, `tools/undistort_videos.py` follows the equivalent inverse convention: it scales the camera intrinsics to the actual video resolution and remaps frames at that size.

Size metadata alone does not describe crop offsets, padding or letterboxing, digital zoom, rotation, or arbitrary warps. Those inputs require an explicit image transform and are not currently supported for resolution-aware projection.

The calibration subsystem is part of the engine layer because [projection](pipeline.md) is part of the run itself. The engine executes against explicit runtime calibration artifacts.

`assets/calibration/camera_correspondences.json` is still available as an auxiliary correspondence source, but the canonical runtime calibration asset is the bundle file above.

Cowbook's calibration math is built on OpenCV's calibration and projection stack. See [References and Attribution](references.md) for the upstream documentation links.
