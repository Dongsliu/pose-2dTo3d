#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
姿势检测器
使用MMPose + RTMW进行2D全身姿势检测

技术挑战记录:
挑战: 从2D图片中准确检测人体关键点
解决方案: 使用RTMW-x模型，支持133个关键点高精度检测
限制: 遮挡情况下精度下降，需要置信度评估
未来改进: 集成多模型融合，处理复杂遮挡场景
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from . import mmcv_compat  # noqa: F401  必须在 import mmpose 之前生效

# 抑制一些不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)

class PoseDetector:
    """
    2D姿势检测器
    使用RTMPose进行全身姿势检测（身体+手部+面部）
    """
    
    # COCO-WholeBody关键点定义（133个点）
    BODY_KEYPOINTS = 17  # 身体关键点
    FOOT_KEYPOINTS = 6   # 脚部关键点
    FACE_KEYPOINTS = 68  # 面部关键点
    HAND_KEYPOINTS = 21  # 每只手的关键点
    
    def __init__(
        self,
        model_name: str = 'rtmw-x',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.3
    ):
        """
        初始化姿势检测器
        
        Args:
            model_name: 模型名称（'rtmw-x' 或 'rtmw-l'）
            device: 计算设备（'cuda' 或 'cpu'）
            confidence_threshold: 置信度阈值
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.config = None
        
        print(f"初始化姿势检测器: {model_name}")
        print(f"设备: {device}")
        
        if device == 'cuda' and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        self._load_model()
    
    def _load_model(self):
        """加载MMPose模型"""
        try:
            from mmpose.apis import init_model
            from mmengine.config import Config
            
            # 获取模型配置路径
            models_dir = Path(__file__).parent.parent / 'models'
            config_path = models_dir / 'configs' / f'{self.model_name}_8xb320-270e_cocktail14-384x288.py'
            checkpoint_path = models_dir / f'{self.model_name}_8xb320-270e_cocktail14-384x288.pth'
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"模型文件不存在: {checkpoint_path}\n"
                    f"请运行: python models/download_models.py"
                )
            
            # 模型文件由启动脚本统一准备，检测阶段不执行下载。
            if not config_path.exists():
                raise FileNotFoundError(
                    f"缺少 RTMW 配置文件: {config_path}\n"
                    "请先重新运行 start.bat，让启动流程下载模型文件。"
                )

            print(f"加载配置: {config_path}")
            print(f"加载模型: {checkpoint_path}")
            # MMPose 1.2 只接受这几个 KLDiscretLoss 参数；不要合并
            # 新版配置中的 label_beta、mask 等未知字段。
            config = Config.fromfile(str(config_path))
            config.model.head.loss = dict(
                type='KLDiscretLoss',
                beta=1.0,
                label_softmax=True,
                use_target_weight=True,
            )
            self.model = init_model(
                config,
                str(checkpoint_path),
                device=self.device,
            )
            
            print("✅ 模型加载成功")
            
        except ImportError as e:
            raise ImportError(
                f"MMPose未安装或版本不兼容: {e}\n"
                f"请运行: pip install setuptools mmcv-lite \"mmpose>=1.1.0,<1.3.0\""
            )
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def detect(
        self,
        image: np.ndarray,
        bbox: Optional[List[float]] = None
    ) -> Dict:
        """
        检测图片中的人体姿势
        
        Args:
            image: 输入图片(BGR格式)
            bbox: 人体边界框 [x, y, w, h]，None则使用全图
            
        Returns:
            检测结果字典，包含:
            - keypoints: 关键点坐标 (N, 2)
            - scores: 置信度 (N,)
            - bbox: 边界框
            - valid: 有效关键点mask (N,)
        """
        if image is None or image.size == 0:
            raise ValueError("输入图片为空")
        
        # 如果没有指定bbox，使用全图
        if bbox is None:
            h, w = image.shape[:2]
            bbox = [0, 0, w, h]
        
        try:
            # 使用MMPose推理
            results = self._inference_mmpose(image, bbox)
            
            # 后处理
            results = self._postprocess(results)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"姿势检测失败: {e}")
    
    def _inference_mmpose(self, image: np.ndarray, bbox: List[float]) -> Dict:
        """使用MMPose进行推理"""
        
        # 禁止使用随机点作为姿态结果。
        if self.model is None:
            raise RuntimeError("RTMW 模型未正确初始化，已拒绝生成伪造姿态")
        
        try:
            from mmpose.apis import inference_topdown
            from mmpose.structures import PoseDataSample
            import mmengine
            
            # 准备输入
            results = inference_topdown(self.model, image)
            
            if len(results) == 0:
                raise ValueError("未检测到人体")
            
            # 提取第一个结果
            result = results[0]
            pred_instances = result.pred_instances
            
            keypoints = pred_instances.keypoints[0]  # (133, 2)
            scores = pred_instances.keypoint_scores[0]  # (133,)
            
            return {
                'keypoints': keypoints,
                'scores': scores,
                'bbox': bbox,
                'valid': scores > self.confidence_threshold
            }
            
        except Exception as e:
            raise RuntimeError(f"MMPose 推理失败: {e}") from e
    
    def _postprocess(self, results: Dict) -> Dict:
        """后处理检测结果"""
        
        # 归一化坐标（可选）
        keypoints = results['keypoints']
        scores = results['scores']
        valid = results['valid']
        
        # COCO-WholeBody: body17 + foot6 + face68 + hand21 + hand21
        body_end = self.BODY_KEYPOINTS
        foot_end = body_end + self.FOOT_KEYPOINTS
        face_end = foot_end + self.FACE_KEYPOINTS
        left_hand_end = face_end + self.HAND_KEYPOINTS
        right_hand_end = left_hand_end + self.HAND_KEYPOINTS
        results['grouped'] = {
            'body': keypoints[:body_end],
            'foot': keypoints[body_end:foot_end],
            'face': keypoints[foot_end:face_end],
            'left_hand': keypoints[face_end:left_hand_end],
            'right_hand': keypoints[left_hand_end:right_hand_end],
        }
        
        # 计算整体置信度
        results['overall_confidence'] = float(np.mean(scores[valid]))
        
        # 统计信息
        results['stats'] = {
            'total_keypoints': len(keypoints),
            'valid_keypoints': int(np.sum(valid)),
            'body_confidence': float(np.mean(scores[:body_end])),
            'face_confidence': float(np.mean(scores[foot_end:face_end])) if len(scores) >= face_end else 0.0,
            'hands_confidence': float(np.mean(scores[face_end:right_hand_end])) if len(scores) >= right_hand_end else 0.0,
        }
        
        return results
    
    def visualize(
        self,
        image: np.ndarray,
        results: Dict,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图片
            results: 检测结果
            show_confidence: 是否显示置信度
            
        Returns:
            可视化图片
        """
        vis_image = image.copy()
        keypoints = results['keypoints']
        scores = results['scores']
        valid = results['valid']
        
        # 绘制关键点
        for i, (kpt, score, is_valid) in enumerate(zip(keypoints, scores, valid)):
            if not is_valid:
                continue
            
            x, y = int(kpt[0]), int(kpt[1])
            
            # 根据置信度设置颜色
            if score > 0.8:
                color = (0, 255, 0)  # 绿色-高置信度
            elif score > 0.5:
                color = (0, 255, 255)  # 黄色-中等置信度
            else:
                color = (0, 0, 255)  # 红色-低置信度
            
            cv2.circle(vis_image, (x, y), 3, color, -1)
            
            if show_confidence:
                cv2.putText(
                    vis_image,
                    f"{score:.2f}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1
                )
        
        # 显示统计信息
        stats = results.get('stats', {})
        info_text = [
            f"Overall: {results.get('overall_confidence', 0):.2f}",
            f"Valid: {stats.get('valid_keypoints', 0)}/{stats.get('total_keypoints', 0)}",
            f"Body: {stats.get('body_confidence', 0):.2f}",
            f"Face: {stats.get('face_confidence', 0):.2f}",
            f"Hands: {stats.get('hands_confidence', 0):.2f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(
                vis_image,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis_image,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
        
        return vis_image
    
    def save_results(self, results: Dict, output_path: str):
        """保存检测结果到JSON文件"""
        import json
        
        # 转换numpy数组为列表
        output_data = {
            'keypoints': results['keypoints'].tolist(),
            'scores': results['scores'].tolist(),
            'bbox': results['bbox'],
            'valid': results['valid'].tolist(),
            'stats': results['stats'],
            'overall_confidence': results['overall_confidence']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存到: {output_path}")


if __name__ == "__main__":
    # 测试代码
    print("姿势检测器模块测试")
    
    detector = PoseDetector(model_name='rtmw-x')
    print("✅ 检测器初始化成功")

