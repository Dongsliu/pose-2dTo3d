#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""让 mmcv-lite 能被 MMPose import。

Torch 2.7 没有官方 MMCV CUDA wheel，因此装的是 mmcv-lite。
MMPose 在 import 时仍会拉 mmcv.ops（EDPose 等用不到的算子），
lite 包没有 mmcv._ext，会直接报错。RTMW 推理本身不需要这些算子。
"""

from __future__ import annotations

import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace


def _missing_op(name):
    def _missing(*args, **kwargs):
        raise RuntimeError(
            f'mmcv CUDA op "{name}" 不可用（当前为 mmcv-lite）。'
            'RTMW 推理不需要该算子。'
        )

    _missing.__name__ = name
    return _missing


class _DummyExt(ModuleType):
    def __init__(self):
        super().__init__('mmcv._ext')
        self.__file__ = __file__
        self.__package__ = 'mmcv'
        self.__path__ = []
        self.__spec__ = importlib.machinery.ModuleSpec(
            'mmcv._ext', loader=None, is_package=False
        )

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _missing_op(name)


def apply():
    if getattr(apply, '_done', False):
        return

    dummy = _DummyExt()
    sys.modules['mmcv._ext'] = dummy

    import mmcv.utils.ext_loader as ext_loader

    ext_loader.load_ext = lambda name, funcs: SimpleNamespace(
        **{fun: _missing_op(fun) for fun in funcs}
    )
    ext_loader.check_ops_exist = lambda: False

    try:
        import mmengine.utils.dl_utils.misc as mmengine_misc
        mmengine_misc.mmcv_full_available = lambda: False
    except Exception:
        pass

    # PyTorch 2.6+ 默认 weights_only=True，OpenMMLab 的 .pth 需要旧行为
    import torch

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_load(*args, **kwargs)

    torch.load = _load

    apply._done = True


apply()
