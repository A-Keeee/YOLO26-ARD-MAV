# ARD-MAV：GLAD 公开方法结构与 YOLO26n

**当前不使用 ByteTrack、DeepSORT 或其他多目标关联器。**
本程序按 GLAD 论文公开的方法结构实现原视频的单目标检测与局部搜索，以 YOLO26n 替换全局、局部 YOLOv5。
当前不能称为“只替换 YOLO 的严格数值复现”，具体缺口见下文。

## 方法结构

当前实现使用以下结构：

- 相对速度卡尔曼状态 (vx, vy, ax, ay)，用于预测搜索窗口中心。
- 论文 Eq.4 的窗口边长 L = 300 + 4*T_lost。
- 连续局部丢失 30 帧后，下一帧回到全局，不进行周期性全图重扫或同帧全局回退。
- 全局、局部置信度分别为论文 Table II 的 T0=0.5、T1=0.1。
- 全局／局部外观与运动检测级联。
- 均匀 30×20 网格 LK 和 RANSAC 单应变换，用于相机运动补偿。

算法来源：[论文正文及公式](https://arxiv.org/html/2312.11008v1)、
[官方代码](https://github.com/WindyLab/Global-Local-MAV-Detection)。

## 每帧执行方式

1. GLOBAL：先运行全局 YOLO26（GAD）；没有有效结果时，运行全局运动检测（GMD）。
2. 目标检测成功后，下一帧进入 LOCAL。先用单应变换补偿相机运动，再叠加相对速度预测搜索中心。
3. LOCAL：在自适应窗口中先运行局部 YOLO26（LAD），失败后运行局部运动检测（LMD）。
   当帧不会同时运行全局 YOLO26。局部失败后继续预测并扩大窗口，连续失败 30 帧后下一帧进入 GLOBAL。
4. 运动检测包括：补偿后的帧差、光照／背景运动阈值修正、形态学处理、连通区域和邻近框合并、
   Shi-Tomasi／LK 运动方向与速度方差筛选，以及 32×32 外观 CNN 的 MAV／clutter 分类。

相对速度的测量为“当前检测中心减去经相机运动补偿的上一帧中心”。
无有效检测时只预测，不用预测位置伪造测量来校正滤波器。
单应估计失败时，运动分支不返回未经补偿的运动候选；ROI 暂用相对速度预测，并在日志中记录估计失败。
局部 YOLO 候选取最高置信度、同分时按预测距离排序；不添加论文未规定的固定 50 像素门限。
局部运动候选取离预测位置最近者。
这是单目标搜索过程，不提供跨目标 ID，也不生成带虚构 ID 的 MOT 文件。

## 仍然无法保证严格一致的部分

- 官方 README 明确表示包含完整卡尔曼滤波、自适应窗口等的代码尚未公开。
  本程序依据论文状态和公式重建；Q/R/P 噪声参数、alpha/beta 修正系数等未公开数值，在 YAML 中明确标为实现选值。
- 官方基础代码与论文也存在差异：运动筛选用了标准差／均值比，且局部分支阈值与论文表格不同；
  本程序优先遵循论文 Eq.14 的方差和 Table II 的阈值，而非逐行照搬基础 demo。
- 当前两个 YOLO26n 权重来自此前的 MuSGD、学习率 0.0054、batch 16、50 轮训练。
  论文检测器使用 Adam、初始学习率 0.01、batch 32、150 轮，
  因此当前训练流程也并非“仅替换 YOLO 版本”。训练配置与启动方法见[仓库首页](../README.md)。
- 保留现有去重后的数据，训练／验证随机划分不保证与未公布随机种子的原实验完全相同。
  正式比较还需使用论文的原视频协议及 IoU=0.5 的 11 点 AP，不能直接比较局部裁剪验证 mAP。

原视频功能试跑中仍观察到失检后的搜索窗口漂移和运动分支误判。
分支和切换测试通过，不代表已达到论文检测精度。

## 文件与模型来源

本目录保留追踪、校准及模型文件，使用环境中安装的 Ultralytics 包，安装方法见[仓库首页](../README.md)：

- scripts/track.py：原视频读取、两级检测切换、相对速度滤波、可视化和日志。
- scripts/motion.py：单应估计、帧差、运动筛选和 CNN 推理。
- scripts/calibrate_kalman.py：使用原视频检测记录和 XML 标注离线校准滤波参数。
- configs/track_glad.yaml：论文公开参数与标明来源的实现选值。
- configs/calibrate_kalman.yaml：校准视频划分、参数搜索范围。
- weights/global_best.pt、weights/local_best.pt：已经训练好的两个 YOLO26n 最佳权重。
- weights/Net_best.pth：作者公开的约 249 KB 外观二分类 CNN，在运动分支中筛选无人机／背景候选。
  它采用两层卷积、三层全连接结构，输入为 32×32 图块；独立于两个 YOLO 检测器，YOLO 训练不会更新它。
  当前仓库直接复用作者权重，未提供该分类器的训练脚本。

分类器下载自
[官方 Net_best.pth](https://github.com/WindyLab/Global-Local-MAV-Detection/blob/main/weights/Net_best.pth)，
SHA-256 为 0d3a4631060339ee72551a89e95887fc49a2d50f9f3473ef354cf3db2830faf5。
根据官方 Functions.py／MOD2.py，保留 OpenCV BGR 通道顺序、缩放到 [0,1]，不做额外标准化，类别 1 为 MAV。
track.py 推理不读取 XML 或 YOLO 标注，不根据真实框初始化目标；calibrate_kalman.py 离线校准会读取 XML 评分。

## 运行

```bash
# 在独立仓库根目录执行，先激活已安装依赖的环境
conda activate ard-mav

python tracking/scripts/track.py --dry-run
python tracking/scripts/track.py --show
python tracking/scripts/track.py --max-frames 300
python tracking/scripts/track.py --source /absolute/path/to/videos/phantom08.mp4 --show
```

也可将 --source 指向原视频目录。每个视频重新初始化搜索状态和滤波器。
配置中的相对路径以配置文件所在目录为准，命令行 --source 的相对路径以当前目录为准。
不会启动任何模型训练。

## 用标注校准卡尔曼参数

```bash
python tracking/scripts/calibrate_kalman.py
```

该命令只运行推理和参数搜索，不训练模型，也不覆盖 track_glad.yaml。
校准配置默认使用 3 个训练视频选择参数，另外 3 个验证视频检查效果；严格检查现有 splits.yaml，拒绝混入测试视频。
max_frames=0 表示完整视频，快速检查可改成 300。

1. 用当前配置运行完整检测级联，记录每帧实际检测框和用于 ROI 预测的单应矩阵。
2. 保持检测记录不变，复用运行时的 RelativeVelocity 滤波器搜索参数。
   **标注只用于计算校正前的预测误差，不作为滤波器的观测输入**；漏检、误检均保留。
   XML 缺失或包含多个目标时，不用于单目标评分，并单独统计；这些帧仍正常传播检测和滤波状态。
3. 训练视频每 30 帧从真实检测后的滤波状态出发，停止接收新观测，依次预测 1、5、15、29 帧。
   每步仍使用该帧图像估计的相机补偿；不输入未来标注位置。按各视频、各跨度的 RMSE 等权平均选择参数，
   避免训练视频自然漏检太少，仅优化连续检测成功时的表现。再在独立验证视频上重放检查。
4. 使用候选参数重新运行验证视频的完整级联，检查窗口改变后的真实检测结果。

当前 Q=qI、R=rI、P0=pI，同步缩放三者不改变滤波增益，因此固定 R=1，搜索 Q/R 和 P0/R。
Q/R 较大通常更快响应变化，也更容易受检测抖动影响；较小通常更平滑，但可能跟不上机动。
P0/R 主要影响刚发现目标时的收敛。它们是本实现的调参比例，不能解释成已准确估计出物理噪声方差。

结果位于 runs/kalman_calibration/，重复执行自动递增：

- baseline_config.yaml、calibration_config.yaml：原追踪配置和本次校准设置。
- baseline_train/、baseline_val/：原配置的完整视频、检测日志和摘要。
- candidate_val/：候选参数重新运行验证视频的结果。
- report.json：全部搜索结果、各跨度预测误差、漏检后误差、ROI 中心覆盖率、IoU≥0.5 检测统计和原检测记录路径。
- candidate_track.yaml：可直接交给 track.py 的候选配置，路径已转为绝对路径。

report.json 中 recommended=true 要求验证重放及完整重跑的预测 RMSE 都降低，且完整重跑后的等权平均召回率、F1 和 ROI 中心覆盖率均不下降。
false 表示本次结果不足以支持替换默认参数，候选配置仍会保留供检查。
确认报告后可运行，例如：

```bash
python tracking/scripts/track.py --config runs/kalman_calibration/candidate_track.yaml --show
```

这里的检测统计是校准子集诊断，不是论文 AP 或最终测试集成绩。
误检会直接污染观测，相机补偿误差也会传入预测，单独调整 Q/R/P0 无法解决所有漂移。
绿色线连接的是原始检测中心，不是滤波后的轨迹；评估滤波效果应查看 ROI、漏检时的橙色预测点及报告误差。

### 已有完整视频校准记录

下列数值来自此前的完整视频校准，运行输出位于本地实验目录，不随源码仓库提供。
重新运行校准命令可生成自己的 report.json 与 candidate_track.yaml；更换环境、模型或数据后应重新检查结果。
训练视频为 phantom84、phantom80、phantom61；验证视频为 phantom04、phantom73、phantom32。
未使用测试视频选参。phantom61 第 482 帧包含两个标注框，按单目标标注歧义排除评分，未改动原 XML。

- 原参数 Q=0.01、R=1、P0=1；训练视频多跨度预测选择 Q=0.001、R=1、P0=0.1。
- 训练视频、预测跨度等权平均的预测 RMSE：12.27 → 10.56 像素。
- 验证视频完整重跑，逐视频等权平均的位置预测 RMSE：915.84 → 316.96 像素。
- 但同一验证子集的平均检测召回率：68.85% → 65.82%；F1：77.88% → 75.35%；ROI 中心覆盖率也下降。

该次报告的 recommended=false，**默认 track_glad.yaml 保持原参数**。
当前仅调三个标量协方差，尚不足以改善完整追踪效果；应继续定位误检观测及漏检时的预测发散。

## 输出

新结果默认保存到仓库根目录的 runs/glad_yolo26/，重复运行自动递增。

- 原视频名.mp4：原始尺寸和帧顺序的检测视频，不保留音轨；显示 GLOBAL/LOCAL、检测分支和丢失帧数。
- 原视频名_frames.csv：每个输入帧一行，包含空检测帧、当前／下一帧模式、实际检测框、ROI、预测中心、ROI 单应矩阵和运动级联统计。
- 原视频名_summary.json：四个检测分支的调用次数和实际检测来源，明确标注未使用 MOT 跟踪器。
- config.yaml：本次参数、版本和权重路径。

预测中心仅用于搜索及橙色标记，不当作真实检测输出。检测覆盖率和处理速度不是准确率或跟踪指标。
