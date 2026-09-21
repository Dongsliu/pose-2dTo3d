#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理多张图片
"""

import argparse
import sys
from pathlib import Path
import cv2
import json
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pose_detector import PoseDetector
from core.pose_lifter import PoseLifter
from core.constraint_solver import ConstraintSolver
from core.image_io import imread

def process_image(image_path, detector, lifter, solver, output_dir):
    """处理单张图片"""
    try:
        # 加载图片
        image = imread(image_path)
        if image is None:
            return {'success': False, 'error': '无法读取图片'}
        
        # 检测
        results_2d = detector.detect(image)
        results_3d = lifter.lift_to_3d(
            results_2d['keypoints'],
            results_2d['scores']
        )
        constrained = solver.apply_constraints(results_3d['keypoints_3d'])
        
        # 保存
        output_data = {
            'input_image': str(image_path.name),
            'keypoints_2d': results_2d['keypoints'].tolist(),
            'keypoints_3d': constrained['keypoints_3d'].tolist(),
            'confidence_2d': results_2d['overall_confidence'],
            'confidence_3d': results_3d['depth_confidence'],
        }
        
        output_file = output_dir / f"{image_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        return {
            'success': True,
            'confidence': results_2d['overall_confidence']
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='批量处理姿势图片')
    parser.add_argument('--input-dir', '-i', required=True, help='输入图片目录')
    parser.add_argument('--output-dir', '-o', required=True, help='输出JSON目录')
    parser.add_argument('--pattern', default='*.jpg', help='文件匹配模式')
    parser.add_argument('--model', default='rtmw-x', help='模型名称')
    parser.add_argument('--device', default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    # 准备目录
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = list(input_dir.glob(args.pattern))
    if not image_files:
        print(f"❌ 未找到匹配的图片: {input_dir}/{args.pattern}")
        return 1
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 确定设备
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 初始化模型
    print(f"初始化模型（设备: {device}）...")
    detector = PoseDetector(model_name=args.model, device=device)
    lifter = PoseLifter(device=device)
    solver = ConstraintSolver(constraint_strength=0.5)
    
    # 批量处理
    print("开始处理...")
    results = []
    
    for image_path in tqdm(image_files, desc="处理进度"):
        result = process_image(image_path, detector, lifter, solver, output_dir)
        results.append({
            'file': image_path.name,
            **result
        })
    
    # 统计
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print("\n" + "=" * 70)
    print("批处理完成")
    print("=" * 70)
    print(f"总计: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    if success_count > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / success_count
        print(f"平均置信度: {avg_confidence:.2f}")
    
    # 保存汇总
    summary_file = output_dir / '_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总已保存到: {summary_file}")
    
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

