#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行姿势检测脚本
从图片中检测姿势并保存为JSON
"""

import argparse
import sys
from pathlib import Path
import cv2
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pose_detector import PoseDetector
from core.hand_refiner import HandRefiner
from core.pose_lifter import PoseLifter
from core.constraint_solver import ConstraintSolver
from core.logger import get_logger
from core.image_io import imread, imwrite

def main():
    parser = argparse.ArgumentParser(description='从图片检测人体姿势')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSON路径')
    parser.add_argument('--model', '-m', default='rtmw-x', choices=['rtmw-x', 'rtmw-l'], help='使用的模型')
    parser.add_argument('--device', '-d', default='auto', choices=['auto', 'cuda', 'cpu'], help='计算设备')
    parser.add_argument('--refine-hands', action='store_true', help='启用手部精细化')
    parser.add_argument('--constraint-strength', type=float, default=0.5, help='物理约束强度(0-1)')
    parser.add_argument('--visualize', '-v', help='保存可视化图片路径')
    
    args = parser.parse_args()
    
    # 初始化日志
    logger = get_logger()
    start_time = logger.log_operation_start("Pose Detection")
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        print(f"❌ 错误: 输入文件不存在: {input_path}")
        return 1
    
    # 确定设备
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"输入: {input_path}")
    logger.info(f"输出: {args.output}")
    logger.info(f"模型: {args.model}")
    logger.info(f"设备: {device}")
    logger.info(f"手部精细化: {'是' if args.refine_hands else '否'}")
    logger.info(f"约束强度: {args.constraint_strength}")
    
    print("=" * 70)
    print("姿势检测")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"输出: {args.output}")
    print(f"模型: {args.model}")
    print(f"设备: {device}")
    print(f"手部精细化: {'是' if args.refine_hands else '否'}")
    print(f"约束强度: {args.constraint_strength}")
    print("=" * 70)
    
    try:
        # 加载图片
        logger.step(1, 5, "加载图片...")
        print("\n[1/5] 加载图片...")
        image = imread(input_path)
        if image is None:
            logger.error("无法读取图片")
            print(f"❌ 错误: 无法读取图片")
            return 1
        logger.debug(f"图片尺寸: {image.shape[1]}x{image.shape[0]}", width=image.shape[1], height=image.shape[0])
        print(f"✅ 图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 2D姿势检测
        logger.step(2, 5, "检测2D姿势...")
        print("\n[2/5] 检测2D姿势...")
        detector = PoseDetector(model_name=args.model, device=device)
        results_2d = detector.detect(image)
        logger.info(f"检测到 {len(results_2d['keypoints'])} 个关键点")
        logger.info(f"整体置信度: {results_2d['overall_confidence']:.2f}")
        logger.debug("2D检测结果", keypoints_count=len(results_2d['keypoints']), confidence=results_2d['overall_confidence'])
        print(f"✅ 检测到 {len(results_2d['keypoints'])} 个关键点")
        print(f"   整体置信度: {results_2d['overall_confidence']:.2f}")
        
        # 手部精细化（可选）
        if args.refine_hands:
            logger.step(3, 5, "手部精细化...")
            print("\n[3/5] 手部精细化...")
            try:
                refiner = HandRefiner()
                hand_results = refiner.refine_hands(
                    image,
                    results_2d.get('grouped', {})
                )
                if hand_results.get('left_hand') is not None:
                    left_conf = hand_results['confidences'].get('left', 0)
                    logger.info(f"左手检测完成: 置信度 {left_conf:.2f}")
                    print(f"   ✅ 左手: 置信度 {left_conf:.2f}")
                if hand_results.get('right_hand') is not None:
                    right_conf = hand_results['confidences'].get('right', 0)
                    logger.info(f"右手检测完成: 置信度 {right_conf:.2f}")
                    print(f"   ✅ 右手: 置信度 {right_conf:.2f}")
            except Exception as e:
                logger.warning(f"手部精细化失败: {e}")
                print(f"   ⚠️  手部精细化失败: {e}")
        else:
            logger.info("跳过手部精细化")
            print("\n[3/5] 跳过手部精细化")
        
        # 2D到3D
        logger.step(4, 5, "重建3D姿势...")
        print("\n[4/5] 重建3D姿势...")
        lifter = PoseLifter(device=device)
        results_3d = lifter.lift_to_3d(
            results_2d['keypoints'],
            results_2d['scores']
        )
        logger.info("3D姿势重建完成")
        logger.info(f"深度置信度: {results_3d['depth_confidence']:.2f}")
        logger.debug("3D重建结果", depth_confidence=results_3d['depth_confidence'])
        print(f"✅ 3D姿势重建完成")
        print(f"   深度置信度: {results_3d['depth_confidence']:.2f}")
        if results_3d.get('warning'):
            logger.warning(results_3d['warning'])
            print(f"   ⚠️  {results_3d['warning']}")
        
        # 物理约束
        logger.step(5, 5, "应用物理约束...")
        print("\n[5/5] 应用物理约束...")
        solver = ConstraintSolver(constraint_strength=args.constraint_strength)
        constrained = solver.apply_constraints(results_3d['keypoints_3d'])
        logger.info(f"约束评分: {constrained['constraint_score']:.2f}")
        logger.debug("约束求解结果", constraint_score=constrained['constraint_score'], violations_count=len(constrained['violations']))
        print(f"✅ 约束评分: {constrained['constraint_score']:.2f}")
        if constrained['violations']:
            logger.warning(f"发现 {len(constrained['violations'])} 个约束违规")
            print(f"   发现 {len(constrained['violations'])} 个约束违规")
        
        # 保存结果
        logger.info("保存结果...")
        print("\n保存结果...")
        output_data = {
            'input_image': str(input_path.absolute()),
            'model': args.model,
            'device': device,
            'keypoints_2d': results_2d['keypoints'].tolist(),
            'keypoints_3d': constrained['keypoints_3d'].tolist(),
            'scores': results_2d['scores'].tolist(),
            'confidence_2d': results_2d['overall_confidence'],
            'confidence_3d': results_3d['depth_confidence'],
            'constraint_score': constrained['constraint_score'],
            'stats': results_2d['stats'],
            'violations': constrained['violations']
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.success(f"结果已保存到: {output_path}")
        print(f"✅ 结果已保存到: {output_path}")
        
        # 可视化（可选）
        if args.visualize:
            logger.info("生成可视化...")
            print("\n生成可视化...")
            vis_image = detector.visualize(image, results_2d)
            imwrite(args.visualize, vis_image)
            logger.success(f"可视化已保存到: {args.visualize}")
            print(f"✅ 可视化已保存到: {args.visualize}")
        
        # 保存日志缓存（用于Blender UI显示）
        logger.info("更新日志缓存...")
        cache_data = {
            'recent_logs': logger.get_recent_logs(20)
        }
        cache_file = project_root / 'logs' / 'ui_cache.json'
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.log_operation_end("Pose Detection", start_time)
        
        print("\n" + "=" * 70)
        print("✅ 完成！")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        print("\n\n⚠️  用户中断")
        return 130
    
    except Exception as e:
        logger.error(f"检测失败: {e}", exc_info=sys.exc_info())
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

