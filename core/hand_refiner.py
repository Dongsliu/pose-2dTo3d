#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手部精细化模块
使用MediaPipe Hands增强手部关键点检测

技术挑战记录:
挑战: 手指有21个关节，需要极高的检测精度
解决方案: 双模型融合（RTMW + MediaPipe Hands），手部区域裁剪放大
限制: 手部遮挡或不在图片中时无法检测
未来改进: 多帧时序平滑，手部遮挡恢复
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class HandRefiner:
    """
    手部精细化检测器
    使用MediaPipe Hands提升手部关键点精度
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        初始化手部检测器
        
        Args:
            max_num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = None
        
        print("初始化手部精细化检测器（MediaPipe Hands）")
        self._load_model()
    
    def _load_model(self):
        """加载MediaPipe Hands模型"""
        try:
            import mediapipe as mp
            
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            print("✅ MediaPipe Hands加载成功")
            
        except ImportError:
            raise ImportError(
                "MediaPipe未安装\n"
                "请运行: pip install mediapipe"
            )
        except Exception as e:
            raise RuntimeError(f"MediaPipe Hands加载失败: {e}")
    
    def refine_hands(
        self,
        image: np.ndarray,
        coarse_hand_keypoints: Optional[Dict] = None
    ) -> Dict:
        """
        精细化手部关键点
        
        Args:
            image: 输入图片(BGR格式)
            coarse_hand_keypoints: RTMW检测的粗略手部关键点
            
        Returns:
            精细化的手部关键点字典
        """
        if self.hands is None:
            raise RuntimeError("MediaPipe Hands未初始化")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 如果有粗略关键点，裁剪手部区域
        if coarse_hand_keypoints:
            refined_results = self._refine_with_roi(image_rgb, coarse_hand_keypoints, w, h)
        else:
            refined_results = self._detect_full_image(image_rgb, w, h)
        
        return refined_results
    
    def _detect_full_image(self, image_rgb: np.ndarray, w: int, h: int) -> Dict:
        """在全图中检测手部"""
        results = self.hands.process(image_rgb)
        
        refined_hands = {
            'left_hand': None,
            'right_hand': None,
            'confidences': {}
        }
        
        if not results.multi_hand_landmarks:
            return refined_hands
        
        # 处理检测到的手
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # 确定左右手
            hand_label = handedness.classification[0].label.lower()  # 'left' or 'right'
            confidence = handedness.classification[0].score
            
            # 提取关键点
            landmarks = np.array([
                [lm.x * w, lm.y * h, lm.z]
                for lm in hand_landmarks.landmark
            ])
            
            refined_hands[f'{hand_label}_hand'] = landmarks
            refined_hands['confidences'][hand_label] = confidence
        
        return refined_hands
    
    def _refine_with_roi(
        self,
        image_rgb: np.ndarray,
        coarse_keypoints: Dict,
        w: int,
        h: int
    ) -> Dict:
        """
        使用ROI裁剪增强检测
        
        Args:
            image_rgb: RGB图片
            coarse_keypoints: 粗略关键点
            w, h: 图片尺寸
        """
        refined_hands = {
            'left_hand': None,
            'right_hand': None,
            'confidences': {}
        }
        
        # 对左右手分别处理
        for hand_name in ['left_hand', 'right_hand']:
            if hand_name not in coarse_keypoints or coarse_keypoints[hand_name] is None:
                continue
            
            hand_kpts = coarse_keypoints[hand_name]
            
            # 计算手部边界框（带扩展）
            x_min = max(0, int(np.min(hand_kpts[:, 0])) - 50)
            x_max = min(w, int(np.max(hand_kpts[:, 0])) + 50)
            y_min = max(0, int(np.min(hand_kpts[:, 1])) - 50)
            y_max = min(h, int(np.max(hand_kpts[:, 1])) + 50)
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # 裁剪手部区域
            hand_roi = image_rgb[y_min:y_max, x_min:x_max]
            
            if hand_roi.size == 0:
                continue
            
            # 在ROI中检测
            results = self.hands.process(hand_roi)
            
            if results.multi_hand_landmarks:
                # 取第一个检测结果
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                
                confidence = handedness.classification[0].score
                
                # 提取关键点并恢复到原图坐标
                roi_w, roi_h = x_max - x_min, y_max - y_min
                landmarks = np.array([
                    [lm.x * roi_w + x_min, lm.y * roi_h + y_min, lm.z]
                    for lm in hand_landmarks.landmark
                ])
                
                refined_hands[hand_name] = landmarks
                refined_hands['confidences'][hand_name.split('_')[0]] = confidence
        
        return refined_hands
    
    def fuse_with_coarse(
        self,
        coarse_keypoints: Dict,
        refined_keypoints: Dict,
        fusion_weight: float = 0.7
    ) -> Dict:
        """
        融合粗略和精细关键点
        
        Args:
            coarse_keypoints: RTMW检测的关键点
            refined_keypoints: MediaPipe精细化的关键点
            fusion_weight: 精细关键点的融合权重（0-1）
            
        Returns:
            融合后的关键点
        """
        fused = {}
        
        for hand_name in ['left_hand', 'right_hand']:
            coarse_hand = coarse_keypoints.get(hand_name)
            refined_hand = refined_keypoints.get(hand_name)
            
            if refined_hand is not None and coarse_hand is not None:
                # 加权融合（只融合x, y坐标）
                fused_xy = (
                    fusion_weight * refined_hand[:, :2] +
                    (1 - fusion_weight) * coarse_hand[:, :2]
                )
                
                # 使用MediaPipe的z坐标
                fused[hand_name] = np.concatenate([
                    fused_xy,
                    refined_hand[:, 2:3]
                ], axis=1)
                
            elif refined_hand is not None:
                fused[hand_name] = refined_hand
            elif coarse_hand is not None:
                fused[hand_name] = coarse_hand
            else:
                fused[hand_name] = None
        
        # 保留置信度信息
        fused['confidences'] = refined_keypoints.get('confidences', {})
        
        return fused
    
    def visualize(
        self,
        image: np.ndarray,
        hand_keypoints: Dict,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        可视化手部关键点
        
        Args:
            image: 原始图片
            hand_keypoints: 手部关键点
            draw_connections: 是否绘制连接线
            
        Returns:
            可视化图片
        """
        vis_image = image.copy()
        
        # MediaPipe手部连接关系
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 大拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 手掌
        ]
        
        # 绘制左右手
        colors = {
            'left_hand': (255, 0, 0),  # 蓝色
            'right_hand': (0, 0, 255)  # 红色
        }
        
        for hand_name, color in colors.items():
            hand = hand_keypoints.get(hand_name)
            if hand is None:
                continue
            
            # 绘制关键点
            for i, kpt in enumerate(hand):
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(vis_image, (x, y), 4, color, -1)
                cv2.circle(vis_image, (x, y), 5, (255, 255, 255), 1)
                
                # 标注关键点索引
                cv2.putText(
                    vis_image,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1
                )
            
            # 绘制连接线
            if draw_connections:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand[start_idx]
                    end = hand[end_idx]
                    cv2.line(
                        vis_image,
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        color,
                        2
                    )
        
        # 显示置信度
        confidences = hand_keypoints.get('confidences', {})
        y_offset = 30
        for hand_type, conf in confidences.items():
            text = f"{hand_type.capitalize()} Hand: {conf:.2f}"
            cv2.putText(
                vis_image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis_image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
            y_offset += 25
        
        return vis_image
    
    def __del__(self):
        """清理资源"""
        if self.hands:
            self.hands.close()


if __name__ == "__main__":
    # 测试代码
    print("手部精细化模块测试")
    
    refiner = HandRefiner()
    print("✅ 手部检测器初始化成功")

