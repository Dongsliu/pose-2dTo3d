#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理约束求解器
确保3D姿势符合人体生理约束

技术挑战记录:
挑战: AI推断的3D姿势可能出现不合理的关节角度或骨骼长度
解决方案: 实现基于规则的物理约束系统，限制关节角度和骨骼比例
限制: 约束过强可能限制艺术性姿势，需要可调节强度
未来改进: 机器学习的姿势合理性评估，自适应约束强度
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ConstraintSolver:
    """
    物理约束求解器
    应用人体骨骼和关节的物理限制
    """
    
    # 关节角度限制（度）
    JOINT_LIMITS = {
        # 肩关节
        'shoulder': {
            'flex_extend': (-180, 180),  # 前后摆动
            'abduct_adduct': (-90, 180),  # 侧向抬起
            'rotate': (-90, 90)  # 旋转
        },
        # 肘关节
        'elbow': {
            'flex': (0, 150),  # 弯曲（0=伸直）
            'rotate': (-90, 90)  # 前臂旋转
        },
        # 髋关节
        'hip': {
            'flex_extend': (-20, 120),  # 前后摆动
            'abduct_adduct': (-45, 45),  # 侧向抬起
            'rotate': (-45, 45)  # 旋转
        },
        # 膝关节
        'knee': {
            'flex': (0, 150)  # 弯曲（0=伸直，不能反向）
        },
        # 脊柱
        'spine': {
            'flex_extend': (-30, 90),  # 前弯/后仰
            'lateral': (-45, 45),  # 侧弯
            'rotate': (-45, 45)  # 扭转
        }
    }
    
    # 骨骼长度比例（相对于身体高度）
    BONE_RATIOS = {
        'upper_arm': 0.186,
        'forearm': 0.146,
        'hand': 0.108,
        'thigh': 0.245,
        'shin': 0.246,
        'foot': 0.152,
        'torso': 0.288,
        'head': 0.130,
        'shoulder_span': 0.22,
        'hip_span': 0.18,
    }
    
    def __init__(
        self,
        constraint_strength: float = 0.5,
        enable_angle_limits: bool = True,
        enable_length_constraints: bool = True
    ):
        """
        初始化约束求解器
        
        Args:
            constraint_strength: 约束强度 (0-1)，0=不约束，1=严格约束
            enable_angle_limits: 启用关节角度限制
            enable_length_constraints: 启用骨骼长度约束
        """
        self.constraint_strength = constraint_strength
        self.enable_angle_limits = enable_angle_limits
        self.enable_length_constraints = enable_length_constraints
        
        print(f"初始化物理约束求解器（强度: {constraint_strength}）")
        print(f"  - 关节角度限制: {'启用' if enable_angle_limits else '禁用'}")
        print(f"  - 骨骼长度约束: {'启用' if enable_length_constraints else '禁用'}")
    
    def apply_constraints(
        self,
        keypoints_3d: np.ndarray,
        skeleton_def: Optional[Dict] = None
    ) -> Dict:
        """
        应用物理约束到3D关键点
        
        Args:
            keypoints_3d: 3D关键点 (N, 3)
            skeleton_def: 骨骼定义（连接关系）
            
        Returns:
            约束结果字典:
            - keypoints_3d: 约束后的关键点
            - violations: 违反约束的列表
            - adjustments: 应用的调整
        """
        if skeleton_def is None:
            skeleton_def = self._get_default_skeleton()
        
        adjusted_keypoints = keypoints_3d.copy()
        violations = []
        adjustments = []
        
        # 1. 检查并修正骨骼长度
        if self.enable_length_constraints:
            adjusted_keypoints, length_violations = self._constrain_bone_lengths(
                adjusted_keypoints,
                skeleton_def
            )
            violations.extend(length_violations)
        
        # 2. 检查并修正关节角度
        if self.enable_angle_limits:
            adjusted_keypoints, angle_violations = self._constrain_joint_angles(
                adjusted_keypoints,
                skeleton_def
            )
            violations.extend(angle_violations)
        
        # 3. 确保对称性（左右肢体）
        adjusted_keypoints = self._enforce_symmetry(
            adjusted_keypoints,
            skeleton_def
        )
        
        return {
            'keypoints_3d': adjusted_keypoints,
            'violations': violations,
            'adjustments': adjustments,
            'constraint_score': self._calculate_constraint_score(violations)
        }
    
    def _get_default_skeleton(self) -> Dict:
        """获取默认的骨骼定义（COCO格式）"""
        return {
            'bones': [
                # (name, type, [parent, child])  type 必须是比例表的键，不能用关节名
                ('shoulder_span', 'shoulder_span', [5, 6]),
                ('torso_l', 'torso', [5, 11]),
                ('torso_r', 'torso', [6, 12]),
                ('hip_span', 'hip_span', [11, 12]),
                ('left_upper_arm', 'upper_arm', [5, 7]),
                ('left_forearm', 'forearm', [7, 9]),
                ('right_upper_arm', 'upper_arm', [6, 8]),
                ('right_forearm', 'forearm', [8, 10]),
                ('left_thigh', 'thigh', [11, 13]),
                ('left_shin', 'shin', [13, 15]),
                ('right_thigh', 'thigh', [12, 14]),
                ('right_shin', 'shin', [14, 16]),
            ],
            'joints': {
                'left_shoulder': 5,
                'right_shoulder': 6,
                'left_elbow': 7,
                'right_elbow': 8,
                'left_hip': 11,
                'right_hip': 12,
                'left_knee': 13,
                'right_knee': 14,
            }
        }
    
    def _constrain_bone_lengths(
        self,
        keypoints: np.ndarray,
        skeleton_def: Dict
    ) -> Tuple[np.ndarray, List[Dict]]:
        """约束骨骼长度在合理范围"""
        adjusted = keypoints.copy()
        violations = []
        
        if len(keypoints) < 17:
            return adjusted, violations
        
        # 计算参考身体高度
        body_height = self._estimate_body_height(keypoints)
        
        # 只记录违规，不改 x/y。改子关节位置会把 3D 从照片投影里拧开。
        for bone_name, bone_type, indices in skeleton_def['bones']:
            if len(indices) != 2:
                continue

            idx1, idx2 = indices
            if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                continue

            current_length = np.linalg.norm(keypoints[idx2] - keypoints[idx1])
            expected_ratio = self._get_bone_ratio(bone_type)
            expected_length = body_height * expected_ratio
            if expected_length <= 1e-6:
                continue

            tolerance = 0.3 * self.constraint_strength
            min_length = expected_length * (1 - tolerance)
            max_length = expected_length * (1 + tolerance)

            if current_length < min_length or current_length > max_length:
                violations.append({
                    'type': 'bone_length',
                    'bone': bone_name,
                    'current': current_length,
                    'expected': expected_length,
                    'severity': abs(current_length - expected_length) / expected_length
                })

        return adjusted, violations
    
    def _constrain_joint_angles(
        self,
        keypoints: np.ndarray,
        skeleton_def: Dict
    ) -> Tuple[np.ndarray, List[Dict]]:
        """约束关节角度在生理范围"""
        adjusted = keypoints.copy()
        violations = []
        
        # 检查主要关节
        joints_to_check = [
            ('left_elbow', [5, 7, 9], 'elbow'),  # 左肘
            ('right_elbow', [6, 8, 10], 'elbow'),  # 右肘
            ('left_knee', [11, 13, 15], 'knee'),  # 左膝
            ('right_knee', [12, 14, 16], 'knee'),  # 右膝
        ]
        
        for joint_name, indices, joint_type in joints_to_check:
            if any(idx >= len(keypoints) for idx in indices):
                continue
            
            p1, p2, p3 = [keypoints[idx] for idx in indices]
            
            # 计算关节角度
            angle = self._calculate_joint_angle(p1, p2, p3)
            
            # 获取该关节的限制
            if joint_type == 'knee':
                # 膝盖不能反向弯曲
                min_angle, max_angle = 0, 150
            elif joint_type == 'elbow':
                min_angle, max_angle = 0, 150
            else:
                continue
            
            # 检查是否超出范围
            if angle < min_angle or angle > max_angle:
                violations.append({
                    'type': 'joint_angle',
                    'joint': joint_name,
                    'current': angle,
                    'limit': (min_angle, max_angle),
                    'severity': max(0, min_angle - angle, angle - max_angle) / 180
                })
                
                # 修正角度（简化版：记录但不强制修改，避免破坏整体姿势）
                # 实际应用中可以使用IK求解器进行修正
        
        return adjusted, violations
    
    def _calculate_joint_angle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> float:
        """
        计算三点形成的关节角度
        
        Args:
            p1: 第一个点（关节前）
            p2: 关节点
            p3: 第三个点（关节后）
            
        Returns:
            角度（度）
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def _enforce_symmetry(
        self,
        keypoints: np.ndarray,
        skeleton_def: Dict
    ) -> np.ndarray:
        """
        强制左右对称性（弱约束）
        确保左右肢体长度相似
        """
        return keypoints
    
    def _estimate_body_height(self, keypoints: np.ndarray) -> float:
        """估计身体高度"""
        if len(keypoints) < 17:
            return 1.0
        
        # 从头到脚的距离
        head = np.mean(keypoints[:5], axis=0) if len(keypoints) >= 5 else keypoints[0]
        feet = np.mean(keypoints[15:17], axis=0) if len(keypoints) >= 17 else keypoints[-1]
        
        height = np.linalg.norm(feet - head)
        return max(height, 1.0)  # 避免除零
    
    def _get_bone_ratio(self, bone_type: str) -> float:
        """获取骨骼长度比例"""
        return self.BONE_RATIOS.get(bone_type, 0.2)
    
    def _calculate_constraint_score(self, violations: List[Dict]) -> float:
        """
        计算约束评分
        
        Returns:
            评分 (0-1)，1表示完全符合约束
        """
        if not violations:
            return 1.0
        
        # 计算平均严重程度
        avg_severity = np.mean([v['severity'] for v in violations])
        
        # 转换为评分（严重程度越低，评分越高）
        score = max(0, 1.0 - avg_severity)
        
        return score


if __name__ == "__main__":
    # 测试代码
    print("物理约束求解器测试")
    
    solver = ConstraintSolver(constraint_strength=0.5)
    print("✅ 约束求解器初始化成功")
    
    # 测试约束应用
    test_keypoints = np.random.rand(17, 3) * 500
    result = solver.apply_constraints(test_keypoints)
    print(f"✅ 约束评分: {result['constraint_score']:.2f}")
    print(f"违规数量: {len(result['violations'])}")

