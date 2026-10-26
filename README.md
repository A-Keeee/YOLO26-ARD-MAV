# ARD-MAV：YOLO26n 全局／局部无人机检测与追踪

基于 ARD-MAV 空对空无人机数据集，训练全局、局部两个 YOLO26n 检测器，并按
[GLAD 论文](https://arxiv.org/abs/2312.11008)的公开方法组织原视频中的单目标搜索。
全局检测负责发现目标，局部检测通过裁剪提高小目标的有效分辨率；检测失败时使用运动检测分支辅助搜索。

本仓库提供数据准备、训练配置、视频推理和卡尔曼参数校准脚本，直接依赖安装的 Ultralytics Python 包。
**仓库根目录就是本 README 所在目录，无需外层 Ultralytics 源码仓库。以下命令均在此目录执行。**

当前版本使用单类别 `Drone`，不接入 ByteTrack／DeepSORT，不输出多目标 ID。
这是基于 GLAD 公开结构的 YOLO26 实现；作者尚未公开完整滤波与搜索代码，现有训练配方也与论文不同，
因此不声称严格复现论文数值结果。方法细节与差异见 [tracking/README.md](tracking/README.md)。

## 目录结构

```text
ARD-MAV/
├── README.md
├── configs/
│   ├── dataset.yaml               # 全局 YOLO 数据路径与类别
│   ├── dataset_local.yaml         # 局部训练／验证数据与整图测试入口
│   ├── prepare.yaml               # 原视频和 VOC XML 转换配置
│   ├── prepare_local.yaml         # 320×320 局部裁剪与背景采样
│   ├── train_global.yaml          # 全局 YOLO26n，默认 50 轮
│   ├── train_local.yaml           # 局部 YOLO26n，默认 50 轮
│   ├── train_baseline.yaml        # 历史整图基线
│   └── resume_baseline.yaml       # 历史基线中断续训模板
├── scripts/
│   ├── prepare.py
│   └── train.py
├── weights/
│   └── yolo26n.pt                 # 官方预训练权重，用于初始化训练
├── tracking/
│   ├── configs/
│   │   ├── track_glad.yaml        # 视频、模型与全局／局部搜索参数
│   │   └── calibrate_kalman.yaml  # 标注辅助的离线滤波参数搜索
│   ├── scripts/
│   │   ├── track.py
│   │   ├── motion.py
│   │   └── calibrate_kalman.py
│   ├── weights/
│   │   ├── global_best.pt         # 本项目训练的全局检测器
│   │   ├── local_best.pt          # 本项目训练的局部检测器
│   │   └── Net_best.pth           # GLAD 作者发布的二分类 CNN
│   └── README.md
├── docs/
│   ├── small_targets.md          # 小目标方法与数据集调查
│   └── dataset_audit/            # 标注检查、重复帧及清理记录
└── runs/                         # 本地生成的训练／视频／校准结果，Git 忽略
```

原始视频、全局 YOLO 数据和局部裁剪数据可放在仓库外，实际位置由 YAML 指定。
数据与实验输出不随普通源码提交提供，下文的实验数量是已有实验记录。

## 环境安装

建议为独立仓库创建 Python 3.10 环境。使用 NVIDIA GPU 时，先按
[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装匹配本机驱动的 `torch`、`torchvision`。
随后安装依赖；以下以 Ultralytics 8.4.183 为安装版本：

```bash
conda create -n ard-mav python=3.10 -y
conda activate ard-mav

# 使用 GPU 时，在这里按 PyTorch 官方说明安装 torch、torchvision
python -m pip install "ultralytics==8.4.183" numpy opencv-python PyYAML
```

如果已经有可运行的 `yolo` 环境，可直接使用。无需在本仓库执行 `pip install -e .`。
检查实际加载的包和 GPU：

```bash
python -c "import ultralytics, torch, cv2; print('Ultralytics:', ultralytics.__version__, ultralytics.__file__); print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
```

当前脚本使用 YOLO26、MuSGD 和 `nms=False`，安装版本需要支持这些接口。
训练和追踪默认 `device: 0`；无 GPU 时可改为 `cpu`。视频显示需要图形桌面，服务器运行保持 `show: false`。

## 数据与路径配置

从 [ARD-MAV 官方项目](https://github.com/WindyLab/Global-Local-MAV-Detection#dataset)获取数据。
原始目录应保留视频与 VOC XML 的对应关系：

```text
<原始数据目录>/
├── videos/phantom05.mp4
└── Annotations/phantom05/phantom05_0001.xml
```

转换后的全局数据包含 `images/{train,val,test}`、`labels/{train,val,test}` 和 `splits.yaml`。
局部数据包含 `images/{train,val}`、`labels/{train,val}`，以及生成的 `manifest.csv`、`summary.yaml`。

**现有 YAML 保留了本机绝对路径。迁移或克隆后，请先按自己的目录修改：**

- [configs/dataset.yaml](configs/dataset.yaml)：`path` 指向转换后的全局 YOLO 数据根目录。
- [configs/dataset_local.yaml](configs/dataset_local.yaml)：`path` 指向局部数据根目录，`test` 指向全局 `images/test`。
- [configs/prepare.yaml](configs/prepare.yaml)：`source` 指向含 `videos/`、`Annotations/` 的原始目录。
- [configs/train_global.yaml](configs/train_global.yaml)、[configs/train_local.yaml](configs/train_local.yaml)：
  将 `model` 改为本机预训练权重路径，或在命令行用 `model=weights/yolo26n.pt` 覆盖。
- [tracking/configs/track_glad.yaml](tracking/configs/track_glad.yaml)：`source` 指向原视频或视频目录；检查三个权重路径。
- [tracking/configs/calibrate_kalman.yaml](tracking/configs/calibrate_kalman.yaml)：`dataset` 指向原始数据，
  `splits` 指向全局数据中的 `splits.yaml`。

例如，全局数据配置可改为：

```yaml
path: /absolute/path/to/ARD-MAV-YOLO
train: images/train
val: images/val
test: images/test
names:
  0: Drone
```

数据集 `path` 建议使用绝对路径，避免准备脚本与 Ultralytics 对相对数据根目录的解释不同。
训练脚本中 `data`、`project` 的相对路径以配置文件目录为基准，`model` 的相对路径以执行命令的工作目录为基准。
追踪配置中的路径以配置文件目录为基准，命令行 `--source` 的相对路径以工作目录为基准。
历史基线／续训配置如果继续使用，也需要更新其中的本机路径。

### 从原始视频转换全局数据

已有已清理的全局数据时可跳过此步。新建数据时，先设置 `prepare.yaml` 的 `source` 和 `dataset.yaml` 的 `path`，再执行：

```bash
python scripts/prepare.py --config configs/prepare.yaml
```

脚本按视频划分训练／验证／测试集，默认保留所有标注帧，并生成 YOLO 标签和 `splits.yaml`。
默认划分为 37／8／15 个视频，15 个测试视频沿用官方名单，其余按 `seed: 0` 划分。
输出目录必须尚不存在，脚本不会覆盖已有数据，也不会启动训练。

原始数据存在 `phantom44` 与 `phantom32` 的重叠片段。
已有全局数据删除了训练集 780 张重复图片及其标签，保留验证集；重新转换会再次生成这些帧，
**转换脚本不自动去重，需按[数据检查报告](docs/dataset_audit/README.md)复核并清理后再生成局部数据。**
当前清理记录为训练 63,933 张、验证 14,447 张、测试 28,337 张；不能将不同清理版本的实验直接比较。

### 生成局部训练／验证数据

确认 `dataset.yaml` 指向已清理的全局数据、`dataset_local.yaml` 指向尚不存在的输出目录后执行：

```bash
python scripts/prepare.py --config configs/prepare_local.yaml
```

默认 `python scripts/prepare.py` 也选择局部准备配置。生成规则：

- 以标注目标中心为参考裁剪 320×320 原图，边缘处平移窗口，并保留裁剪内所有可见标注。
- 空标注源图生成背景块；正样本源图有 25% 概率额外采样一个与所有标注框无交集的背景块。
- 训练时再将局部图缩放到网络输入尺寸 640，裁剪尺寸与网络输入尺寸分别配置。
- 保持源视频划分不变；`manifest.csv` 记录来源与原图坐标，`summary.yaml` 记录生成数量及设置。

已有局部数据为训练 79,666 张、验证 18,021 张，其中背景分别为 16,337／3,834 张。
局部验证集根据真实标注选取 ROI，**只用于监控局部模型训练，不代表完整视频的检测精度**。
没有生成由真实框定位的测试裁剪，`dataset_local.yaml` 的 `test` 使用原始整图测试集。

## 训练两个 YOLO26n 检测器

全局与局部检测器均为独立的标准 YOLO26n，默认从 `weights/yolo26n.pt` 初始化，
网络输入 640、batch 16、训练 50 轮，优化器为 MuSGD。
全局模型使用整图，局部模型使用前述裁剪；全局 `mosaic: 0.5`，局部 `mosaic: 0.0`。

先检查配置，不加载权重、不开始训练：

```bash
python scripts/train.py --config configs/train_global.yaml model=weights/yolo26n.pt --dry-run
python scripts/train.py --config configs/train_local.yaml model=weights/yolo26n.pt --dry-run
```

确认后自行启动，建议在同一 GPU 上依次执行：

```bash
python scripts/train.py --config configs/train_global.yaml model=weights/yolo26n.pt
python scripts/train.py --config configs/train_local.yaml model=weights/yolo26n.pt
```

命令行 `key=value` 覆盖 YAML，例如：

```bash
python scripts/train.py --config configs/train_local.yaml model=weights/yolo26n.pt batch=8 epochs=50
```

默认输出为 `runs/yolo26n_global_50e/` 和 `runs/yolo26n_local_50e/`，重复实验自动追加数字后缀。
每轮更新 `weights/last.pt`，最佳验证指标更新 `weights/best.pt`；当前配置每 10 轮另存周期权重。
`results.csv`、`args.yaml` 和曲线用于检查实际训练过程。

中断续训使用尚未完成且保留优化器状态的 `last.pt`，例如：

```bash
python scripts/train.py --config configs/train_local.yaml model=runs/yolo26n_local_50e/weights/last.pt resume=True epochs=100 batch=16
```

这里 `epochs=100` 表示包含已训练轮数的总预算。已完成训练且剥离优化器状态的权重不能按中断检查点恢复；
更换全局／局部模型或数据准备方式时，应开启独立实验。

## 权重说明：Net_best.pth 是什么？

本项目使用四个不同用途的权重文件：

- `weights/yolo26n.pt`：Ultralytics 官方预训练 YOLO26n，用于初始化全局、局部训练。
- `tracking/weights/global_best.pt`：本项目训练的全局 YOLO26n。
- `tracking/weights/local_best.pt`：本项目训练的局部 YOLO26n。
- **`tracking/weights/Net_best.pth`：GLAD 作者发布的外观二分类 CNN，不是 YOLO 检测器或卡尔曼模型。**

`Net_best.pth` 的网络是 LeNet 风格的两层卷积、三层全连接网络，输入为 32×32 候选图块，
输出背景／干扰物（类别 0）与无人机（类别 1）。网络结构对应
[官方 Functions.py 中的 Net](https://github.com/WindyLab/Global-Local-MAV-Detection/blob/main/Functions.py)。

YOLO 未检出目标后，运动分支通过帧差和运动特征产生候选，再交给该 CNN 过滤背景干扰。
它不负责在整幅图中搜索框，也不进行目标 ID 关联。当前由 `motion.py` 加载，保留 OpenCV BGR 通道顺序，
缩放到 [0,1] 后推理；启动完整追踪流程需要此文件，不能用 YOLO 的 `best.pt` 替换。

该权重直接复用[作者发布的文件](https://github.com/WindyLab/Global-Local-MAV-Detection/blob/main/weights/Net_best.pth)，
未在本项目中重新训练。运行 `scripts/train.py` 只训练 YOLO，不会更新这个分类器；当前仓库未提供其训练脚本。
文件名区分大小写，应为 `Net_best.pth`。

本地文件为 248,841 字节，SHA-256：

```text
0d3a4631060339ee72551a89e95887fc49a2d50f9f3473ef354cf3db2830faf5
```

如缺失，可从上述官方文件页面下载，放入 `tracking/weights/`。
重新训练 YOLO 后，需要把实际实验目录中的 `best.pt` 更新到追踪目录，或修改追踪配置中的权重路径。
下面以没有数字后缀的实验目录为例：

```bash
cp runs/yolo26n_global_50e/weights/best.pt tracking/weights/global_best.pt
cp runs/yolo26n_local_50e/weights/best.pt tracking/weights/local_best.pt
```

## 原视频检测与单目标追踪

设置好 `tracking/configs/track_glad.yaml` 中的视频和模型路径后执行：

```bash
# 仅检查配置和文件是否存在
python tracking/scripts/track.py --dry-run

# 先处理前 300 帧
python tracking/scripts/track.py --max-frames 300

# 处理完整视频并显示；按 q 结束当前视频
python tracking/scripts/track.py --show

# 指定其他视频，或将 --source 指向原视频目录
python tracking/scripts/track.py --source /absolute/path/to/videos/phantom08.mp4 --show
```

流程为全局 YOLO（GAD）→失败时全局运动检测（GMD）；发现目标后切换到局部 YOLO（LAD）→失败时局部运动检测（LMD）。
局部搜索中心由相机运动补偿与相对速度卡尔曼滤波预测，窗口边长为 `300 + 4 × 连续丢失帧数`，
连续局部丢失 30 帧后，下一帧回到全局搜索。实际视频推理不读取真实标注来初始化或修正位置。

结果默认位于 `runs/glad_yolo26/`，重复运行自动递增，包含标注视频、逐帧 CSV、统计 JSON 和配置快照。
白框为局部搜索窗口，橙色点为漏检时的预测中心，绿色线连接最近最多 30 个成功检测中心。
绿色线受到原始检测抖动和相机运动影响，不是滤波后的平滑轨迹。输出视频不保留音轨。

## 卡尔曼参数校准与效果评估

修改 `tracking/configs/calibrate_kalman.yaml` 中的原始数据、视频划分和参数范围后运行：

```bash
python tracking/scripts/calibrate_kalman.py
```

该命令只进行推理与离线参数搜索，不训练 YOLO，也不覆盖默认追踪配置。
使用训练视频选择参数，验证视频检查效果，拒绝使用测试视频调参。
标注仅用于计算预测误差；滤波器接收的仍是实际检测结果，包括误检与漏检。
搜索包含 1、5、15、29 帧无新观测的预测，并对候选参数重新运行完整视频级联。

默认输出到 `runs/kalman_calibration/`，其中 `report.json` 记录误差与检测统计，
`candidate_track.yaml` 是可交给追踪脚本的候选配置。候选不自动生效，具体判断规则见 [追踪说明](tracking/README.md#用标注校准卡尔曼参数)。
已有校准未得到整体效果稳定提升的参数，因此默认仍保留 `Q=0.01、R=1、P0=1`。

评估时需区分局部裁剪验证、整图验证和完整原视频结果。
已有全局模型最佳轮次的验证 mAP50／mAP50-95 为 0.2494／0.0999；局部模型在真值裁剪验证集上为 0.8748／0.5351。
二者输入和评估条件不同，不能据此声称完整视频检测达到 87.48%。
最终比较应在独立测试视频上执行完整搜索流程，以原图坐标评估；当前追踪摘要中的检测覆盖率、处理速度均不是准确率。
论文使用的 IoU=0.5、11 点 AP 也不能直接与 Ultralytics 验证 mAP 混用。

## 仓库内容与引用

`.gitignore` 排除数据目录、实验输出和 Python 缓存。源码、配置、检查文档与四个运行／初始化权重分别保存在上述目录。
数据从官方来源单独获取，实验输出在本机生成；无需将完整 Ultralytics 源码复制进本仓库。

算法与数据来源为 [GLAD 官方项目](https://github.com/WindyLab/Global-Local-MAV-Detection)，
检测框架为 [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)。
上游代码、数据及预训练权重的许可和使用条件以各自项目说明为准。
使用 ARD-MAV 数据或 GLAD 方法开展研究时，请引用原工作：

```bibtex
@article{guo2023globallocal,
  title={Global-Local MAV Detection under Challenging Conditions based on Appearance and Motion},
  author={Hanqing Guo and Ye Zheng and Yin Zhang and Zhi Gao and Shiyu Zhao},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```
