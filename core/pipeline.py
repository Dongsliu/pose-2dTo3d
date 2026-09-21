#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""姿势检测管线，供命令行和预览服务复用。"""

from __future__ import annotations

import numpy as np

from .constraint_solver import ConstraintSolver
from .hand_refiner import HandRefiner
from .pose_detector import PoseDetector
from .pose_lifter import PoseLifter

_CACHE = {}


def _device(device):
    if device and device != 'auto':
        return device
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_pipeline(model_name='rtmw-x', device='auto'):
    resolved = _device(device)
    key = (model_name, resolved)
    if key not in _CACHE:
        _CACHE[key] = {
            'device': resolved,
            'detector': PoseDetector(model_name=model_name, device=resolved),
            'lifter': PoseLifter(device=resolved),
            'refiner': None,
        }
    return _CACHE[key]


def detect_image(
    image,
    model_name='rtmw-x',
    device='auto',
    refine_hands=False,
    constraint_strength=0.5,
    input_image='',
):
    pipe = get_pipeline(model_name, device)
    detector = pipe['detector']
    lifter = pipe['lifter']
    resolved = pipe['device']

    results_2d = detector.detect(image)
    if refine_hands:
        if pipe['refiner'] is None:
            pipe['refiner'] = HandRefiner()
        try:
            pipe['refiner'].refine_hands(image, results_2d.get('grouped', {}))
        except Exception:
            pass

    results_3d = lifter.lift_to_3d(results_2d['keypoints'], results_2d['scores'])
    solver = ConstraintSolver(constraint_strength=constraint_strength)
    constrained = solver.apply_constraints(results_3d['keypoints_3d'])

    def as_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    return {
        'input_image': input_image,
        'model': model_name,
        'device': resolved,
        'keypoints_2d': as_list(results_2d['keypoints']),
        'keypoints_3d': as_list(constrained['keypoints_3d']),
        'scores': as_list(results_2d['scores']),
        'confidence_2d': float(results_2d['overall_confidence']),
        'confidence_3d': float(results_3d['depth_confidence']),
        'method_3d': results_3d['method'],
        'constraint_score': float(constrained['constraint_score']),
        'stats': results_2d['stats'],
        'violations': constrained['violations'],
    }
