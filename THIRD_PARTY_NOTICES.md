# Third-Party Notices

This file is the consolidated notice for third-party components and model
assets used or referenced by this project.

The MIT License in `LICENSE` applies to the original project code and its
modifications. It does not replace the licenses of third-party software,
model weights, external tools, or user-provided assets.

## MMPose, RTMPose/RTMW and SimpleBaseline3D

This project uses MMPose APIs, configuration data, and model checkpoints
published by OpenMMLab.

The RTMPose/RTMW configuration is downloaded from MMPose at runtime and kept
in the local `models/configs/` directory. It is not included in the Git
distribution of this repository.

- Project: MMPose
- License: Apache License 2.0
- Copyright: OpenMMLab

The Apache License 2.0 text is available from the Apache Software Foundation.
Redistributions must retain applicable copyright, attribution, patent, and
NOTICE information from the original distribution.

Model checkpoints remain subject to the terms published by their original
authors and any applicable training-dataset restrictions. The checkpoints
are downloaded separately and are not licensed under this repository's MIT
License.

## MediaPipe

- Purpose: optional hand-keypoint refinement
- License: Apache License 2.0
- Copyright: Google LLC

## PyTorch

- Purpose: model inference
- License: BSD-3-Clause
- Copyright: PyTorch contributors

## Other Python Dependencies

MMCV, MMEngine, OpenCV, NumPy, SciPy, Pillow, PyYAML, Matplotlib, and other
packages installed from `requirements.txt` retain their respective licenses.

## Blender

Blender is invoked as an external command-line tool for FBX export. Blender
is not distributed in this repository and retains its GNU GPL license.

## Assets

Images placed in `samples/poses/` are local, optional user assets and are not
part of the Git distribution. Users are responsible for ensuring that they
have permission to use and distribute those assets.
