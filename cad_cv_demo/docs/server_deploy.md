# 远程下载 SketchGraphs 与服务器部署指南

## 1. SketchGraphs 数据下载

数据形态（见[官方文档](https://princetonlips.github.io/SketchGraphs/)）：

| 形态 | 体积 | 说明 |
|---|---|---|
| 原始 JSON | 128 个 tar.zst 分卷，共 ~43GB | 按卷独立下载，取 1–2 卷即可（约 0.3–0.7GB/卷） |
| 构建序列 | 单文件 ~15GB | 自定义二进制格式，官方基线训练用 |
| 过滤切分 | train / test / validation | 官方过滤子集，体积更小 |

直链位置：打开 <https://princetonlips.github.io/SketchGraphs/>，各条目下的
“available for download here” 即为文件直链（本机 DNS 受限未能解析，未代为核实，
下载前在浏览器打开确认一次）。

服务器下载（Ubuntu 示例）：

```bash
sudo apt install -y aria2 zstd
mkdir -p datasets/sketchgraphs && cd datasets/sketchgraphs

# 下载（aria2 多线程；wget -c 支持断点续传）
aria2c -x16 -s16 "<分卷URL>"          # 或: wget -c "<分卷URL>"

# 解压
zstd -d <分卷>.tar.zst                # 得到 <分卷>.tar
tar -xf <分卷>.tar
```

解析草图 JSON 可使用官方库：`pip install sketchgraphs`
（其 `sketchgraphs.pipeline` 模块含 `make_sketch_dataset` 等解析函数）。
集成路径：草图 JSON → 提取闭合轮廓（线/圆弧/圆）→ 折线采样 → 复用
`cad_cv_demo/dl_raster.py` 的栅格化，作为 `dl_matcher` 的训练类与
`dl_detector` 的贴图形状库。

## 2. 服务器环境

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# CPU 版 torch：
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 有 NVIDIA GPU 时改用对应 CUDA 源，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install ezdxf ezdwg opencv-python-headless numpy matplotlib
```

**重要风险**：`ezdwg 0.9.0` 目前仅确认提供 Windows 轮子（cp310-abi3-win_amd64），
Linux 轮子有无未验证。部署前先在服务器执行：

```bash
pip download ezdwg --no-deps -d /tmp/ezdwg_check
```

- 成功 → 直接可用。
- 失败（无 Linux 轮子）→ 两条替代路径：
  1. 在 Windows 侧用 `cad_cv_demo.shape_match_diagnostics.primitives_from_dwg`
     把 DWG 图元导出为 JSON 中间格式，再传服务器处理；
  2. 输入统一改为 DXF（`ezdxf` 为纯 Python，各平台可用）。

## 3. 部署本项目

```bash
# 本地仓库目录（vendor/、vendor313/、models/ 已在 .gitignore 中，不会进 git）
rsync -av --exclude vendor --exclude vendor313 user@server:/opt/CADGraphChange/
```

- **仅推理**：本地训练后，把 `cad_cv_demo/models/matcher.pt` 与 `detector.pt`
  拷到服务器同路径，CPU 即可运行：
  ```bash
  python cad_cv_demo/cad_cv_pipeline.py --matcher dl \
      --source test.dwg --reference-dir . --output-dir cad_cv_demo/output
  ```
  torch 或权重缺失时管线自动回退启发式匹配，不中断。
- **服务器训练**：
  ```bash
  nohup python cad_cv_demo/train_dl_models.py > train.log 2>&1 &
  tail -f train.log   # 训练循环每约 2% 步输出 step/loss/it/s/ETA
  ```

## 4. 防火墙/代理提示

若服务器出境受限，可在本地下载分卷后 `scp` 上传，或配置
`https_proxy` 环境变量后再用 aria2/wget。
