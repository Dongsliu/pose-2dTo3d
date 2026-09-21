# pose_2dTo3d

一个本地运行的单张图片姿态处理工具：从图片中检测人体 2D 关键点，使用 3D 姿态模型推断身体深度，并在浏览器中预览结果。项目还可以使用本地 Blender 模板导出静态姿态 FBX。

## 功能范围

- 使用 MMPose 的 RTMW 模型从图片中检测 COCO-WholeBody 关键点，最多输出 133 个 2D 关键点。
- 使用 MMPose pose lifter 和 SimpleBaseline3D 权重，根据身体 17 个 2D 关键点推断 3D 身体姿态。单张图片中的前后深度不是直接测量值，而是模型推断结果。
- 对 3D 身体关键点执行基于规则的骨骼长度和关节角度检查，并输出违规列表和约束评分。
- 提供本地 Web 查看器，可显示 2D 关键点、3D 姿态、置信度和约束检查结果。
- 可选地调用 Blender，将检测结果映射到 `product/PoseRig.blend` 中的骨架并导出静态 FBX。

项目不承诺从单张图片恢复真实世界尺度、完整遮挡区域或准确的关节旋转。输出结果应作为姿态估计和后续人工调整的基础。

## 快速开始

1. 创建 Python 虚拟环境并安装依赖。
2. 准备 PyTorch 和 MMPose 依赖。
3. 下载模型权重。首次运行 `start.bat` 时也会自动检查并下载缺少的模型文件。
4. 启动 Web 查看器，上传图片并开始检测。

## 安装

当前依赖说明主要按 Windows + Python 3.11 编写。其他 Python 版本可能可以运行，但未在本项目中验证。

### 1. 创建虚拟环境

```powershell
python --version
python -m venv pose_transfer_env
.\pose_transfer_env\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以直接使用虚拟环境中的 Python，或改用命令提示符：

```bat
pose_transfer_env\Scripts\activate.bat
```

`start.bat` 会自动查找项目目录下的 `.venv` 或 `pose_transfer_env`。如果使用其他环境名称，请手动运行 Web 服务脚本。

### 2. 安装 PyTorch

有 NVIDIA GPU 时，可以根据本机驱动和 CUDA 兼容性选择对应的 PyTorch 安装源。例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

没有 NVIDIA GPU 时可以安装 CPU 版本：

```bash
pip install torch torchvision torchaudio
```

### 3. 安装其余依赖

当前依赖安装前需要先单独准备 `chumpy`：

```bash
pip install wheel setuptools
pip install chumpy --no-build-isolation
pip install -r requirements.txt
```

### 4. 下载模型

模型权重没有提交到 Git 仓库。检测功能必须准备模型文件；可以手动下载：

```bash
python models/download_models.py
```

也可以直接运行 `start.bat`，启动脚本会在启动 Web 服务前检查并下载缺少的模型文件。

模型下载脚本会准备 SimpleBaseline3D 权重、RTMW 权重和 RTMW-x 配置文件。默认流程使用 `rtmw-x`。命令行虽然保留了 `rtmw-l` 选项，但当前自动下载流程没有准备完整的 RTMW-l 配置，因此不属于开箱即用的推荐选项。

## Web 使用

### Windows

双击 `start.bat`，或在已经激活虚拟环境的终端中运行：

```bash
python scripts/view_pose.py --host 0.0.0.0
```

然后在本机打开：

```text
http://127.0.0.1:8765/
```

默认 Web 服务功能包括：

- 拖入图片并调用本地检测接口。
- 拖入已有的 `pose.json` 进行离线预览。
- 查看 2D 关键点和模型推断的 3D 身体姿态。
- 查看 2D 置信度、3D 深度置信度、有效关键点数量和约束评分。
- 拖动 3D 画布旋转视角。

使用 `--host 0.0.0.0` 时，同一局域网的其他设备可以通过运行电脑的 IPv4 地址和端口 `8765` 访问，但仍受操作系统防火墙和网络隔离规则影响。

## 命令行使用

从图片检测姿态并保存 JSON：

```bash
python scripts/detect_pose.py --input image.jpg --output pose.json
```

常用选项：

```bash
python scripts/detect_pose.py \
  --input image.jpg \
  --output pose.json \
  --model rtmw-x \
  --device auto \
  --constraint-strength 0.5 \
  --visualize vis.jpg
```

- `--model`：默认 `rtmw-x`。
- `--device`：`auto`、`cuda` 或 `cpu`。
- `--constraint-strength`：规则检查使用的约束强度参数，范围为 `0` 到 `1`。它影响违规判断阈值，不会把姿态自动修正为标准姿势。
- `--visualize`：可选的 2D 检测可视化输出路径。
- `--refine-hands`：调用项目中的 MediaPipe 手部检测模块。当前实现不会把该模块的结果融合回最终 `pose.json`，因此不能把它视为主流程中的手部精修结果。

## FBX 导出

Web 页面中的“导出 FBX”功能需要同时满足以下条件：

- 项目中存在未随仓库分发的 `product/PoseRig.blend`。
- 本机安装 Blender，或通过 `POSE_BLENDER_EXE` 指定 Blender 可执行文件。
- 模板中存在可用的 Armature；优先查找名为 `PoseRig` 的骨架。

PowerShell 示例：

```powershell
$env:POSE_BLENDER_EXE = "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
python scripts/view_pose.py --host 127.0.0.1
```

导出脚本会把检测到的关节点转换为模板骨骼的头尾位置，然后导出一个静态 FBX。当前实现不是动画重定向系统，也不会生成关键帧动画或完整的关节旋转动画。

## 实际处理流程

```text
[输入图片]
    |
    v
[RTMW 模型推理]
    |  2D 全身关键点，最多 133 点
    v
[SimpleBaseline3D / pose lifter 模型推理]
    |  根据身体 17 个 2D 点推断 3D 深度
    v
[规则检查]
    |  骨骼长度、关节角度、违规列表和评分
    v
[Web 预览或 JSON 输出]
    |
    +--> [可选 Blender 模板映射]
              |
              +--> [静态 FBX]
```

### 哪些部分是模型推理

- `core/pose_detector.py`：加载 RTMW，并调用 MMPose 的 top-down 推理接口检测 2D 关键点。
- `core/pose_lifter.py`：加载 SimpleBaseline3D，并调用 MMPose pose lifter 根据 2D 身体关键点推断 3D 位置和深度。

### 哪些部分不是模型推理

- `core/constraint_solver.py`：基于规则计算骨骼长度和关节角度违规，不是物理模拟器，也不会保证自动修正姿态。
- `scripts/export_pose_fbx.py`：将 3D 关键点映射到 Blender 模板骨骼并导出静态 FBX。
- `viewer/index.html`：负责 2D/3D 结果显示和交互。

## 技术限制

### 单张图片的深度歧义

单张 2D 图片缺少真实的前后深度信息。3D pose lifter 会根据训练数据和 2D 身体关键点推断深度，但以下情况可能产生明显误差：

- 身体或四肢被遮挡。
- 人物处于极端、侧身或训练分布之外的姿势。
- 图片模糊、分辨率低或人物不完整。
- 手脚关键点过小，或背景与人物对比度不足。

项目中的 `confidence_3d` 是根据 2D 关键点置信度计算的估计指标，不是经过独立测试集验证的 3D 误差或准确率。仓库目前没有提供可支持固定百分比准确率的评测数据，因此不对 3D 深度准确率作百分比承诺。

### 使用建议

- 使用清晰、完整且人物占画面比例较高的图片。
- 尽量减少人体遮挡和运动模糊。
- 正面、侧面和轻微斜侧姿态都可以尝试，但复杂姿势需要人工检查。
- 将 3D 结果和约束违规列表作为参考，不要把约束评分当作真实姿态准确率。

## 项目结构

```text
pose-2dTo3d/
├── core/
│   ├── pose_detector.py       # RTMW 2D 姿态检测
│   ├── pose_lifter.py         # SimpleBaseline3D 2D-to-3D 推理
│   ├── constraint_solver.py   # 规则检查和约束评分
│   ├── hand_refiner.py        # 可选 MediaPipe 手部检测模块
│   └── pipeline.py            # Web/管线入口
├── models/
│   └── download_models.py     # 模型下载脚本
├── scripts/
│   ├── detect_pose.py         # 命令行检测入口
│   ├── view_pose.py           # 本地 Web 服务
│   └── export_pose_fbx.py     # Blender FBX 导出脚本
├── viewer/
│   └── index.html             # 浏览器查看器
├── requirements.txt
├── start.bat                  # Windows 启动脚本
└── THIRD_PARTY_NOTICES.md
```

模型权重、生成的配置、上传文件、日志和导出文件默认不会提交到 Git 仓库，具体忽略规则见 `.gitignore`。

## 许可证和第三方组件

本项目原创代码采用 [MIT License](LICENSE)。该许可证不替代第三方代码、模型权重、外部工具或素材各自的许可证。

完整的第三方组件、模型权重和素材说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本项目使用或参考了以下公开项目：

- [MMPose](https://github.com/open-mmlab/mmpose) - 2D/3D 姿态推理框架，Apache-2.0
- [RTMPose / RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - 全身关键点检测模型与配置，Apache-2.0
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) - 可选手部检测模块，Apache-2.0
- [PyTorch](https://github.com/pytorch/pytorch) - 深度学习推理，BSD-3-Clause
- [Blender](https://www.blender.org/) - 外部 FBX 导出工具，GNU GPL
- [blender-pose-2dTo3d](https://github.com/Yxm-bot/blender-pose-2dTo3d/) - 项目起始实现和流程参考

SimpleBaseline3D 和 RTMW 模型权重从 OpenMMLab 公开地址单独下载。模型权重及其训练数据集的使用限制以原始发布方条款为准，不由本项目 MIT 许可证重新授权。
