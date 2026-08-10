# InstantNuRec 推理能力与输入说明

## 一句话概括

InstantNuRec 会将 **NCore V4 格式的、带标定与位姿的自动驾驶多相机行车记录**，通过一次前馈推理重建成静态的 [3D Gaussian Splatting（3DGS）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 场景，并导出为 PLY 文件。

它适合快速预览道路场景的三维重建，或为后续 NuRec 高保真精修提供初始化。

## 此仓库实际输出的内容

本仓库的独立 CLI 只会输出 **静态 Gaussian 层**：

- 三维位置（position）
- 旋转（rotation）
- 尺度（scale）
- 密度/不透明度（density / opacity）
- RGB 与球谐颜色特征（SH）
- 法线（normal）
- 道路与天空语义标记（如可用）

输出是 3DGS PLY，不是通用点云；建议使用 SuperSplat 或 NuRec 容器中的 `ply_viewer` 查看。

> **重要边界**：论文中描述的完整研究模型还包括动态 3DGS、天空 cubemap 与每相机 ISP 校正；本仓库不会导出这些内容。代码中的动态层和天空仅作为合并流程兼容的占位数据。动态物体会尽量从静态 PLY 中排除，而不会作为可运动对象导出。

## 它能做什么

1. **重建静态道路环境**：从连续多帧、多视角观测中恢复道路、建筑、植被、路边设施等的 Gaussian 场景。
2. **利用多相机提升覆盖范围**：可使用前向相机，或同时使用前、左、右等相机。
3. **处理动态目标区域**：模型预测语义；若输入含有 NCore 的动态 cuboid tracks，会结合目标轨迹进一步识别动态区域，并将其从静态输出中移除。
4. **分块处理长片段**：按时间把序列拆为多个 chunk 分别重建，或合并为一个 PLY。
5. **压缩 Gaussian 数量**：启用 `--merge` 后，会对合并结果做视锥归属筛选和 KL 最优体素化，默认目标约为 200 万个静态 Gaussian。
6. **作为后续 NuRec 初始化**：导出的 PLY 可输入更慢、但更高保真的 NuRec 每场景优化流程。

## 它不能直接做什么

- 不能将任意单张图片、普通视频或未标定相机视频直接转换为 3D。
- 不能从此 CLI 得到可随时间运动的车辆、行人等动态 Gaussian 层。
- 不能从此 CLI 得到天空 cubemap、ISP 校正结果或最终 USD/USDZ。
- 不能在 CPU 上推理；需要可用的 CUDA/NVIDIA GPU。
- 不是训练或微调入口；该仓库只实现预训练权重的预测和 PLY 导出。

## 需要怎样的输入？

### 1. 输入入口：NCore V4 序列元数据

命令行的 `--ncore-path` 只接受以下两种形式：

| 形式 | 内容 | 用途 |
| --- | --- | --- |
| 单个 `.json` | 一个 NCore V4 sequence metadata 文件 | 推理一个行车序列 |
| 单个 `.lst` | 每行一个 `.json` 路径 | 批量推理多个序列 |

`.lst` 中的路径可以是绝对路径、相对于 `.lst` 文件所在目录的路径，或以 `~/` 开头的路径；空行和以 `#` 开头的行会被忽略。

### 2. `.json` 所引用的数据

`.json` 不是图像本身。它应当指向可由 NCore V4 loader 打开的序列数据，其中至少应具备：

- **相机图像帧**：每个所选相机的一段连续帧序列。
- **相机标定**：内参、外参和相机模型参数；输入管线也处理 FTheta / OpenCV fisheye 等非针孔模型。
- **车辆（rig）到世界坐标系的时变位姿**：用于把各帧观测放到统一三维坐标系。
- **帧时间戳**：用于按时间采样连续帧，并建立 rolling-shutter 射线的时间关系。

以下数据并非“生成静态 PLY”的核心图像输入，但会提高动态物体处理的效果：

- **cuboid tracks / 3D 目标轨迹**：用于更可靠地识别移动目标并从静态层剔除。
- **相机有效区域 / mask**：用于过滤鱼眼边缘等无效像素。

因此，不能仅创建一个 JSON 文件并放入 JPG/PNG 路径来替代 NCore 序列；文件结构、传感器数据和位姿必须符合 NCore V4 loader 的预期。

### 3. 相机与帧数契约

选择的 checkpoint 决定可接受的相机组合、帧数和推理分辨率：

| `--model` | 默认相机输入 | 每个相机帧数 | 推理分辨率 | Gaussian 头 |
| --- | --- | ---: | ---: | --- |
| `pa-front` | `camera_front_wide_120fov` | 18 | 784 × 448 | 稠密、像素对齐 |
| `pa-multiview` | 前向广角 + 左交叉 + 右交叉 | 18 | 504 × 280 | 稠密、像素对齐 |
| `pq-front` | `camera_front_wide_120fov` | 18 | 784 × 448 | 选择性 point-query，输出较少 |

`pa-multiview` 可通过重复 `--camera-id` 传入 1、3 或 5 个相机；`pa-front` 只能使用 1 个相机；`pq-front` 固定为前向广角相机。相机 ID 必须在 NCore 序列中真实存在。

输入图像会先进行保持宽高比的缩放和中心裁剪，以匹配表中的固定分辨率。模型会校验最终的总视图数是否等于“相机数 × 18”。

### 4. 时间范围与分块

一个 chunk 最多使用每相机 18 帧，单 chunk 覆盖的最长时间约为 13.5 秒。默认 `--max-chunks 8`，因此默认最多覆盖约 108 秒；更长的序列会被截断并打印警告。

提高 `--max-chunks` 能覆盖更长片段，但会增加运行时间和显存需求。

## 推理数据流

```text
NCore V4 sequence JSON
  └─> 图像、标定、相机射线、时间戳、rig 位姿、可选 cuboid tracks
      └─> 以 18 帧/相机进行时间分块与图像预处理
          └─> ViT encoder + DPT / point-query decoder
              └─> 深度、颜色、法线、语义、Gaussian 参数
                  └─> 通过“相机射线 + 深度”恢复三维 Gaussian 中心
                      └─> 剔除天空、ego、动态区域与低密度 Gaussian
                          └─> 每 chunk 导出 PLY，或合并/体素化后导出一个 PLY
```

## 最小运行方式

```bash
python run_inference.py \
  --model pa-front \
  --ncore-path /path/to/sequence.json \
  --output-dir /path/to/output \
  --merge
```

不加 `--merge` 时，输出目录中每个时间 chunk 会生成一个 PLY；加上 `--merge` 后，每个序列会生成一个合并后的 PLY。

输出路径形式如下：

```text
<output-dir>/<run-id>/ply/<sequence-id>/<sequence-id>.ply
```

未合并时，文件名会带有 `_chunk<N>.ply` 后缀。

## 运行前检查清单

- 已安装 Python 3.11 环境和项目依赖。
- `torch.cuda.is_available()` 返回 `True`，并有足够 NVIDIA GPU 显存。
- 已取得与所选 `--model` 一致的权重；首次运行可从 Hugging Face 下载，也可通过 `INSTANT_NUREC_FULL_PT` 指定本地权重。
- NCore sequence `.json` 及它引用的数据文件在本机可访问。
- 输入序列包含所选的 `--camera-id`。
- 对长序列，按需要提高 `--max-chunks`；对显存紧张情况，优先使用 `pa-front` 或较少的多视角相机数。

## 对应实现位置

- CLI 参数、分块和合并行为：`instant_nurec/cli.py`
- 输入序列与相机配置：`instant_nurec/config_schema/dataset.py`
- NCore 数据加载：`instant_nurec/datasets/instantnurec_ncore.py`
- 预训练模型档位：`instant_nurec/pretrained.py`
- 核心前馈重建：`instant_nurec/model/static_core.py`、`instant_nurec/model/inference.py`
- PLY 导出：`instant_nurec/predict/export_ply.py`
