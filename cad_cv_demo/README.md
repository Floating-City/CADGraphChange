# CAD 图纸计算机视觉处理 Demo

本 Demo 针对当前任务实现了：DWG/DXF 读取、零件轮廓视觉分割、1:1 比例恢复、开放轮廓修复、周长计算、物料编码命名、DXF R2004 输出及图像化质检。

## 运行

```powershell
$env:PYTHONPATH="..\vendor;.."
python .\cad_cv_pipeline.py --source ..\test.dwg --reference-dir .. --output-dir .\output
python .\validate_outputs.py .\output\dxf
```

也可以在本目录运行：

```powershell
.\run_demo.ps1
```

## 深度学习检测与匹配（默认启用）

匹配模型有两个实现：`--matcher dl`（默认，深度学习）与 `--matcher heuristic`（原手工特征）。
DL 模式由两部分组成：U-Net 分割检测器在整图栅格切片上分离零件轮廓像素并给出包围框，
CNN 嵌入匹配器（InfoNCE 自监督）以余弦距离匹配参考轮廓，包围框/周长误差超过 15% 的组合直接否决。

训练模型权重（系统 Python 3.13，已装 torch；ezdxf/ezdwg 见 requirements.txt）：

```powershell
python .\train_dl_models.py --source ..\test.dwg --reference-dir ..
python .\cad_cv_pipeline.py --matcher dl --source ..\test.dwg --reference-dir .. --output-dir .\output
```

权重与训练可视化输出在 `cad_cv_demo/models/`。torch 或权重缺失时自动回退启发式匹配
（`run_demo.ps1` 使用的内置 3.12 运行时即走回退路径）。

## 输出

- `output/dxf/`：按 `物料编码_L轮廓周长mm.dxf` 命名的 R2004 文件。
- `output/previews/`：每个结果的渲染预览。
- `output/dwg_cv_detection_overlay.png`：在源 DWG 上的轮廓检测叠加图。
- `output/output_contact_sheet.png`：全部结果的图像化总览。
- `output/results.csv`、`results.json`：比例、尺寸、周长、闭合修复和匹配分数。
- `output/validation_report.json`：版本、图层、闭合性、尺寸实体和文件名周长复核。
- `output/技术报告.md`：技术选型、实现路径、挑战与处理明细。

## 方法说明

算法不是简单复制实体，而是将 CAD 几何栅格化用于视觉验证，并结合端点连通图、尺寸比例主峰、包围框、周长、实体类型和分段长度签名完成识别。对源 DWG 中无法形成高置信独立轮廓的物料，使用所提供的单零件 DXF 执行同样的闭合检查和 1:1 归一化，保证全部物料都有可核验输出。
