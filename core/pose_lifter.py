#!/usr/bin/env python
"""Single-image 2D-to-3D body pose lifting."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from . import mmcv_compat  # noqa: F401  Must run before importing MMPose APIs.


class PoseLifter:
    """Lift COCO body keypoints with MMPose SimpleBaseline3D."""

    H36M_TO_COCO = {
        0: 9,   # nose/head base
        5: 11,  # left shoulder
        6: 14,  # right shoulder
        7: 12,  # left elbow
        8: 15,  # right elbow
        9: 13,  # left wrist
        10: 16, # right wrist
        11: 4,  # left hip
        12: 1,  # right hip
        13: 5,  # left knee
        14: 2,  # right knee
        15: 6,  # left ankle
        16: 3,  # right ankle
    }

    def __init__(
        self,
        model_type: str = "simplebaseline3d",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_type = model_type
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        import mmpose
        from mmpose.apis import init_model

        models_dir = Path(__file__).parent.parent / "models"
        config_path = (
            Path(mmpose.__file__).parent
            / ".mim"
            / "configs"
            / "body_3d_keypoint"
            / "image_pose_lift"
            / "h36m"
            / "image-pose-lift_tcn_8xb64-200e_h36m.py"
        )
        checkpoint_path = (
            models_dir / "simple3Dbaseline_h36m-f0ad73a4_20210419.pth"
        )
        missing = [
            str(path)
            for path in (config_path, checkpoint_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing 3D pose model files: " + ", ".join(missing)
            )
        return init_model(
            str(config_path),
            str(checkpoint_path),
            device=self.device,
        )

    def lift_to_3d(
        self,
        keypoints_2d: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        camera_params: Optional[Dict] = None,
    ) -> Dict:
        keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
        if keypoints_2d.ndim != 2 or keypoints_2d.shape[1] != 2:
            raise ValueError(
                f"Expected keypoints shaped (N, 2), got {keypoints_2d.shape}"
            )
        if len(keypoints_2d) < 17:
            raise ValueError("At least 17 COCO body keypoints are required")

        keypoints_3d = self._model_lift(
            keypoints_2d, confidence, camera_params
        )
        return {
            "keypoints_3d": keypoints_3d,
            "depth_confidence": self._estimate_depth_confidence(confidence),
            "method": self.model_type,
            "warning": None,
        }

    def _model_lift(
        self,
        keypoints_2d: np.ndarray,
        confidence: Optional[np.ndarray],
        camera_params: Optional[Dict],
    ) -> np.ndarray:
        from mmengine.structures import InstanceData
        from mmpose.apis.inference_3d import (
            convert_keypoint_definition,
            inference_pose_lifter_model,
        )
        from mmpose.structures import PoseDataSample

        body_2d = keypoints_2d[:17]
        if confidence is None:
            body_scores = np.ones(17, dtype=np.float32)
        else:
            body_scores = np.asarray(confidence[:17], dtype=np.float32)

        detector_points = np.concatenate(
            [body_2d, body_scores[:, None]], axis=1
        )[None, ...]
        h36m_points = convert_keypoint_definition(
            detector_points, "coco", "h36m"
        )

        min_xy = body_2d.min(axis=0)
        max_xy = body_2d.max(axis=0)
        sample = PoseDataSample()
        sample.pred_instances = InstanceData(
            keypoints=h36m_points[..., :2],
            bboxes=np.array(
                [[min_xy[0], min_xy[1], max_xy[0], max_xy[1]]],
                dtype=np.float32,
            ),
        )
        sample.gt_instances = InstanceData()
        sample.track_id = 0

        if camera_params:
            image_size = (
                int(camera_params.get("width", max_xy[0] + 1)),
                int(camera_params.get("height", max_xy[1] + 1)),
            )
        else:
            image_size = (int(max_xy[0] + 1), int(max_xy[1] + 1))

        results = inference_pose_lifter_model(
            self.model,
            [[sample]],
            with_track_id=True,
            image_size=image_size,
            norm_pose_2d=True,
        )
        if not results:
            raise RuntimeError("SimpleBaseline3D returned no pose")

        lifted = np.asarray(
            results[0].pred_instances.keypoints, dtype=np.float64
        ).squeeze()
        if lifted.shape != (17, 3):
            raise RuntimeError(
                f"Unexpected SimpleBaseline3D output shape: {lifted.shape}"
            )

        # Reorder to X, depth, vertical. Keep X in image-left/image-right
        # order; the MMPose demo flips it only for its own 3D camera view.
        lifted = lifted[..., [0, 2, 1]]
        lifted[..., 2] *= -1

        torso_2d = np.linalg.norm(
            (body_2d[5] + body_2d[6]) * 0.5
            - (body_2d[11] + body_2d[12]) * 0.5
        )
        torso_3d = np.linalg.norm(lifted[8] - lifted[0])
        depth_scale = torso_2d / max(float(torso_3d), 1e-6)

        output = np.zeros((len(keypoints_2d), 3), dtype=np.float64)
        output[:, :2] = keypoints_2d
        # Keep the detector's image-plane projection exact. A monocular pose
        # lifter estimates depth, but its predicted X/Y is not guaranteed to
        # reproject onto the source image and can visibly bend straight limbs.
        pelvis_3d = lifted[0].copy()
        for coco_index, h36m_index in self.H36M_TO_COCO.items():
            joint = (lifted[h36m_index] - pelvis_3d) * depth_scale
            output[coco_index, 2] = joint[1]

        self._regularize_limb_depths(output, lifted)
        self._dampen_frontal_depth(output)

        output[1:5, :2] = (
            output[0, :2]
            + keypoints_2d[1:5]
            - keypoints_2d[0]
        )
        output[1:5, 2] = output[0, 2]
        if len(output) > 17:
            output[17:min(20, len(output)), :2] = (
                output[15, :2]
                + keypoints_2d[17:min(20, len(output))]
                - keypoints_2d[15]
            )
            output[17:min(20, len(output)), 2] = output[15, 2]
        if len(output) > 20:
            output[20:min(23, len(output)), :2] = (
                output[16, :2]
                + keypoints_2d[20:min(23, len(output))]
                - keypoints_2d[16]
            )
            output[20:min(23, len(output)), 2] = output[16, 2]
        if len(output) > 23:
            output[23:min(91, len(output)), :2] = (
                output[0, :2]
                + keypoints_2d[23:min(91, len(output))]
                - keypoints_2d[0]
            )
            output[23:min(91, len(output)), 2] = output[0, 2]
        if len(output) > 91:
            output[91:min(112, len(output)), :2] = (
                output[9, :2]
                + keypoints_2d[91:min(112, len(output))]
                - keypoints_2d[91]
            )
            output[91:min(112, len(output)), 2] = output[9, 2]
        if len(output) > 112:
            output[112:min(133, len(output)), :2] = (
                output[10, :2]
                + keypoints_2d[112:min(133, len(output))]
                - keypoints_2d[112]
            )
            output[112:min(133, len(output)), 2] = output[10, 2]
        return output

    def _regularize_limb_depths(
        self,
        output: np.ndarray,
        lifted: np.ndarray,
    ) -> None:
        """Make paired limb segments share a plausible 3D bone length.

        A monocular lifter can assign a large depth delta to a fully visible
        limb, especially for dance poses outside Human3.6M's training set.
        The detector projection gives a hard lower bound for every bone, and
        the matching left/right segment gives a useful person-specific length
        prior. Keep the model's front/back sign, but solve the magnitude from
        that shared length instead of copying the unstable raw magnitude.
        """
        paired_segments = (
            ((5, 7), (6, 8)),    # upper arms
            ((7, 9), (8, 10)),   # forearms
            ((11, 13), (12, 14)),  # thighs
            ((13, 15), (14, 16)),  # shins
        )

        for pair in paired_segments:
            projected_lengths = [
                float(np.linalg.norm(output[child, :2] - output[parent, :2]))
                for parent, child in pair
            ]
            target_length = max(projected_lengths)

            for (parent, child), projected_length in zip(
                pair, projected_lengths
            ):
                depth_magnitude = np.sqrt(max(
                    target_length * target_length
                    - projected_length * projected_length,
                    0.0,
                ))
                parent_h36m = self.H36M_TO_COCO[parent]
                child_h36m = self.H36M_TO_COCO[child]
                model_delta = (
                    lifted[child_h36m, 1] - lifted[parent_h36m, 1]
                )
                direction = -1.0 if model_delta < 0 else 1.0
                output[child, 2] = (
                    output[parent, 2] + direction * depth_magnitude
                )

    @staticmethod
    def _dampen_frontal_depth(output: np.ndarray) -> None:
        """Prefer an image-plane pose when the body clearly faces front.

        In a frontal view, shorter projected limbs are often caused by the
        pose or detector placement rather than by motion toward the camera.
        Shoulder/hip levelness and visible body width provide a conservative
        frontality estimate. This prevents a lateral yoga bend from becoming
        a forward bend while leaving oblique and profile poses to the lifter.
        """
        shoulder_span = float(np.linalg.norm(output[5, :2] - output[6, :2]))
        hip_span = float(np.linalg.norm(output[11, :2] - output[12, :2]))
        shoulder_center = (output[5, :2] + output[6, :2]) * 0.5
        hip_center = (output[11, :2] + output[12, :2]) * 0.5
        torso_length = max(
            float(np.linalg.norm(shoulder_center - hip_center)), 1.0
        )

        shoulder_tilt = abs(output[5, 1] - output[6, 1]) / max(
            shoulder_span, 1.0
        )
        hip_tilt = abs(output[11, 1] - output[12, 1]) / max(
            hip_span, 1.0
        )
        mean_tilt = (shoulder_tilt + hip_tilt) * 0.5
        mean_width_ratio = (shoulder_span + hip_span) * 0.5 / torso_length

        level_score = np.clip((0.45 - mean_tilt) / 0.30, 0.0, 1.0)
        width_score = np.clip(
            (mean_width_ratio - 0.18) / 0.20, 0.0, 1.0
        )
        frontality = float(level_score * width_score)
        if frontality <= 0.0:
            return

        pelvis_depth = float((output[11, 2] + output[12, 2]) * 0.5)
        depth_factor = 1.0 - 0.92 * frontality
        output[:17, 2] = (
            pelvis_depth
            + (output[:17, 2] - pelvis_depth) * depth_factor
        )

    @staticmethod
    def _estimate_depth_confidence(
        confidence_2d: Optional[np.ndarray],
    ) -> float:
        if confidence_2d is None:
            return 0.65
        scores = np.asarray(confidence_2d[:17], dtype=np.float64)
        if scores.size == 0:
            return 0.65
        if float(scores.max()) > 1.5:
            scores = scores / 10.0
        return float(np.clip(scores.mean() * 0.8, 0.0, 0.8))

    def refine_depth(
        self,
        keypoints_3d: np.ndarray,
        depth_adjustments: Dict[int, float],
    ) -> np.ndarray:
        adjusted = np.asarray(keypoints_3d, dtype=np.float64).copy()
        for index, offset in depth_adjustments.items():
            if 0 <= index < len(adjusted):
                adjusted[index, 2] += offset
        return adjusted

    def normalize_pose(
        self, keypoints_3d: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        keypoints_3d = np.asarray(keypoints_3d, dtype=np.float64)
        center = keypoints_3d.mean(axis=0)
        body_height = np.linalg.norm(
            keypoints_3d[:5].mean(axis=0)
            - keypoints_3d[11:13].mean(axis=0)
        )
        scale = 100.0 / body_height if body_height > 0 else 1.0
        return (keypoints_3d - center) * scale, {
            "center": center,
            "scale": scale,
        }
