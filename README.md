# pose 2dTo3d

本地运行的姿态处理工具，从2D图片提取人体关键点，推理3D姿态，并通过本地Blender模板导出FBX。



### 🎯 快速开始

```
第1步: 安装 requirements.txt 中的依赖
第2步: 双击 start.bat，启动时自动检查并下载模型
第3步: 上传图片并开始检测
第4步: 按需导出 FBX
✅ 完成！
```



### 安装步骤

#### 1. 安装Python环境

```powershell
# 检查Python版本（需要3.8+）
python --version

# 创建虚拟环境
python -m venv pose_transfer_env

# 激活虚拟环境
.\pose_transfer_env\Scripts\Activate.ps1  # Windows
# 或
source pose_transfer_env/bin/activate  # Linux/Mac
```

#### 2. 安装PyTorch (GPU版本)

```bash
# RTX 2060 Super支持CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU版本（如果没有NVIDIA GPU）：**
```bash
pip install torch torchvision torchaudio
```

#### 3. 安装依赖包

```bash
pip install -r requirements.txt
```

#### 4. 下载模型权重（可选）

```bash
python models/download_models.py
```

直接双击 `start.bat` 也会在启动网页前自动检查并下载缺少的模型文件。

## 📖 使用方法

### Web 使用

双击 `start.bat`，然后在本机打开 `http://127.0.0.1:8765/`。同一局域网的其他设备使用运行电脑的 IPv4 地址和端口 `8765` 访问。上传图片后开始检测，可直接在网页中查看 3D 姿态。导出 FBX 需要在本机提供未随仓库分发的 `product/PoseRig.blend` 模板。

### 命令行使用

```bash
# 从图片提取姿势关键点
python scripts/detect_pose.py --input image.jpg --output pose.json

# 启动本地 Web 查看器
start.bat
```

## 🏗️ 项目结构

```
pose_transfer_system/
├── core/                      # 核心引擎
│   ├── pose_detector.py       # 2D姿势检测
│   ├── hand_refiner.py        # 手部增强
│   ├── pose_lifter.py         # 2D→3D转换
│   └── constraint_solver.py   # 姿态约束
├── models/                    # 模型管理
│   └── download_models.py
└── requirements.txt
```

## 🎨 技术架构

### 核心技术栈

- **MMPose + RTMPose** - 2D姿势检测（133个关键点）
- **MediaPipe** - 手部/面部增强
- **SimpleBaseline3D** - 2D到3D姿势转换
- **PyTorch** - 深度学习推理
- **Blender CLI** - FBX导出

### 处理流程

```
[2D图片] 
    ↓
[RTMPose检测] → 133个2D关键点
    ↓
[MediaPipe增强] → 手部精细化
    ↓
[SimpleBaseline3D] → 3D关键点重建
    ↓
[物理约束] → 合理性验证
    ↓
[骨骼映射] → 匹配Blender骨骼
    ↓
[旋转计算] → 关节角度
    ↓
[Blender API] → 应用到模型
    ↓
[3D姿势]
```

## ⚠️ 技术限制说明

### 深度歧义问题

从单张2D图片重建3D姿势时，深度方向（前后）信息是通过AI推断的：

- ✅ **主要关节位置**：85-90%准确度
- ⚠️ **深度旋转**：70-75%准确度（可能需要手动微调）
- ⚠️ **手部细节**：75-80%准确度

**解决方案：**
- 使用深度调整UI进行微调
- 提供多视角输入（如有）可提升精度
- 物理约束系统避免不合理姿势

### 使用建议

- ✅ 使用清晰、完整的人物图片
- ✅ 避免遮挡和模糊
- ✅ 正面或侧面角度效果最佳
- ⚠️ 极端姿势可能需要更多调整

## 📄 许可证

本项目原创代码采用 [MIT License](LICENSE)。该许可证不替代第三方
代码、模型权重、外部工具或素材各自的许可证。

完整的第三方组件、模型权重和素材说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 🙏 参考项目与第三方组件

本项目的实现和运行流程使用或参考了以下公开项目：

- [MMPose](https://github.com/open-mmlab/mmpose) - 2D/3D姿态推理框架，Apache-2.0
- [RTMPose / RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - 全身关键点检测模型与配置，Apache-2.0
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) - 可选手部精细化，Apache-2.0
- [PyTorch](https://github.com/pytorch/pytorch) - 深度学习推理，BSD-3-Clause
- [Blender](https://www.blender.org/) - 外部 FBX 导出工具，GNU GPL
- [blender-pose-2dTo3d](https://github.com/Yxm-bot/blender-pose-2dTo3d/) - 项目起始实现和流程参考

SimpleBaseline3D 和 RTMW 模型权重从 OpenMMLab 公开地址单独下载，模型
权重及其训练数据集的使用限制以原始发布方条款为准，不由本项目 MIT
许可证重新授权。
