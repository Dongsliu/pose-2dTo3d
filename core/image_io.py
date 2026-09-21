#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Windows 下 OpenCV 无法直接读写含中文的路径，改走内存编解码。"""

from pathlib import Path

import cv2
import numpy as np


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or '.jpg'
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
