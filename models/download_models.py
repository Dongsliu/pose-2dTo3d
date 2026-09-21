#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型自动下载脚本
下载所有必需的预训练模型权重
"""

import os
import urllib.request
import sys
from pathlib import Path

# 模型配置
MODELS = {
    'simplebaseline3d': {
        'url': 'https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
        'filename': 'simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
        'size_mb': 17,
        'description': 'SimpleBaseline3D single-image pose lifter'
    },
    'rtmw-x': {
        # 官方权重文件名已变更，本地仍保存为检测器期望的文件名
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
        'filename': 'rtmw-x_8xb320-270e_cocktail14-384x288.pth',
        'size_mb': 100,
        'description': 'RTMW-x 全身姿势检测模型（推荐）'
    },
    'rtmw-l': {
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
        'filename': 'rtmw-l_8xb320-270e_cocktail14-384x288.pth',
        'size_mb': 60,
        'description': 'RTMW-l 全身姿势检测模型（轻量级）'
    }
}

# RTMW配置文件
RTMW_CONFIG = {
    'rtmw-x': {
        'url': 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-x_8xb320-270e_cocktail14-384x288.py',
        'filename': 'rtmw-x_8xb320-270e_cocktail14-384x288.py',
        'description': 'RTMW-x 配置文件'
    }
}

def download_file(url, filepath, description):
    """下载文件并显示进度"""
    print(f"\n下载: {description}")
    print(f"URL: {url}")
    print(f"保存到: {filepath}")
    
    if os.path.exists(filepath):
        print(f"✅ 文件已存在，跳过下载")
        return True
    
    try:
        def reporthook(count, block_size, total_size):
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r下载进度: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, reporthook=reporthook)
        print(f"\n✅ 下载完成")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("模型自动下载脚本")
    print("=" * 70)
    
    # 创建models目录
    models_dir = Path(__file__).parent
    models_dir.mkdir(exist_ok=True)
    
    # 创建configs目录
    configs_dir = models_dir / 'configs'
    configs_dir.mkdir(exist_ok=True)
    
    print(f"\n模型将保存到: {models_dir}")
    print(f"配置文件将保存到: {configs_dir}")
    
    # 下载模型
    print("\n" + "=" * 70)
    print("第1步: 下载模型权重")
    print("=" * 70)
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        filepath = models_dir / model_info['filename']
        if download_file(model_info['url'], filepath, model_info['description']):
            success_count += 1
    
    # 下载配置文件
    print("\n" + "=" * 70)
    print("第2步: 下载配置文件")
    print("=" * 70)
    
    for config_name, config_info in RTMW_CONFIG.items():
        filepath = configs_dir / config_info['filename']
        download_file(config_info['url'], filepath, config_info['description'])
    
    # 创建本地配置文件
    print("\n" + "=" * 70)
    print("第3步: 创建本地配置")
    print("=" * 70)
    
    create_local_configs(models_dir, configs_dir)
    
    # 总结
    print("\n" + "=" * 70)
    print("下载完成总结")
    print("=" * 70)
    print(f"成功下载: {success_count}/{total_count} 个模型")
    
    if success_count == total_count:
        print("\n✅ 所有模型下载成功！")
        print("\n下一步:")
        print("1. 测试模型: python tests/test_pose_detection.py")
        print("2. 在Blender中安装插件")
        return True
    else:
        print("\n⚠️  部分模型下载失败")
        print("请检查网络连接或手动下载")
        return False

def create_local_configs(models_dir, configs_dir):
    """创建本地配置文件"""
    
    # model_config.yaml
    model_config_path = models_dir / 'model_config.yaml'
    model_config_content = f"""# 模型配置文件
# 自动生成，请勿手动修改模型路径

models:
  # 2D姿势检测模型
  pose_2d:
    rtmw-x:
      name: "RTMW-x"
      config: "{configs_dir / 'rtmw-x_8xb320-270e_cocktail14-384x288.py'}"
      checkpoint: "{models_dir / 'rtmw-x_8xb320-270e_cocktail14-384x288.pth'}"
      precision: "high"
      speed: "fast"
      keypoints: 133
      recommended: true
      
    rtmw-l:
      name: "RTMW-l"
      config: "{configs_dir / 'rtmw-l_8xb320-270e_cocktail14-384x288.py'}"
      checkpoint: "{models_dir / 'rtmw-l_8xb320-270e_cocktail14-384x288.pth'}"
      precision: "medium"
      speed: "very_fast"
      keypoints: 133
      recommended: false

  # 手部检测（MediaPipe，无需下载）
  hands:
    mediapipe:
      name: "MediaPipe Hands"
      precision: "high"
      speed: "fast"
      keypoints: 21
      per_hand: true

# 默认模型选择
default:
  pose_2d: "rtmw-x"
  hands: "mediapipe"
"""
    
    with open(model_config_path, 'w', encoding='utf-8') as f:
        f.write(model_config_content)
    
    print(f"✅ 创建配置文件: {model_config_path}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

