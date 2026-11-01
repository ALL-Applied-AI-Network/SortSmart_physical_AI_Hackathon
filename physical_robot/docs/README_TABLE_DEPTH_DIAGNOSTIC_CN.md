# 真实机械臂已知高度盒子诊断 v03

本版本在 v02 空桌验证通过后，测量一个尺子实测高度为 **55--56 mm**
的平整盒子。它用于验证 table depth 能否稳定测出真实障碍物高度。

## 安全行为

- `ACTION = "joints"`，每次只原样发送 observation 中的当前关节和夹爪状态；
- `DEPTH = ["table"]`，只申请 table RGB 和 table depth；
- 不加载 YOLO，不扫描、不下降、不平移、不改变夹爪状态；
- 采集 20 帧，每帧同时拟合盒子顶面和盒子周围的局部桌面；
- 若关节或夹爪状态缺失，直接拒绝发送替代姿态。

## 盒子摆放

把刚才验证过的白色盒子平放在 table camera 画面中央，位置尽量与上一轮
照片一致：位于机械臂正前方，周围留出桌面，不要让机械爪、碗或杂物遮挡。
盒子应完全闭合，顶面不要放置其他物体。

程序只在以下 robot base 区域寻找盒子：

```text
x: 0.24 -- 0.44 m
y: -0.14 -- 0.09 m
候选高度: 35 -- 75 mm
```

如果终端持续显示 `box_height=None`，不要移动机器人；先停止程序，再把盒子
移回画面中央后重试。

## 安装

```bash
cd /path/to/PhysicalAI-Hackathon-2026

cp policy.py policy_before_box_height_v03.py
tar -xzf ../physical_box_height_diagnostic_v03.tar.gz
```

## 先运行 mock

```bash
uv run run.py --mock --trials 1 --trial-s 8
```

mock 没有真实盒子，因此看到 `box_height=None` 是正常的。关键是最后应显示：

```text
✓ Ready for the real robot
```

## 真机运行

v03 与 v02 的请求接口完全相同：table RGB、table depth、joint action。
如果当前已批准的 request 仍有 rollout，可以继续使用；如果已经结束，再执行：

```bash
uv run run.py --submit
```

获得批准后运行：

```bash
uv run run.py
```

采集过程中终端应出现类似：

```text
[depth] sample 1/20: valid=0.923, table_error=+0.9 mm, box_height=53.6, box_valid=True
```

## 查看结果

```bash
RUN=$(ls -dt table_depth_diagnostics/rollout_* | head -1)
echo "$RUN"
cat "$RUN/box_height_summary.json"
```

重点字段：

```text
box_height_frames
box_detection_fraction
box_height_median_mm
box_height_std_mm
box_height_peak_to_peak_mm
maximum_top_plane_rmse_mm
object_height_validation_passed
```

通过门槛：

- 至少 10 帧；
- 至少 90% 帧成功测到盒子；
- 高度中位数落在 55--56 mm 加减 5 mm 的标定容差内；
- 20 帧高度标准差不超过 2 mm；
- 顶面拟合 RMSE 不超过 3 mm；
- `object_height_validation_passed = true`。

当前单帧真实数据用本算法复算得到约 **53.62 mm**，顶面拟合 RMSE 约
**0.79 mm**，符合预期。v03 的目的就是确认这一结果在 20 帧中是否稳定。

## 发回结果

```bash
RUN=$(ls -dt table_depth_diagnostics/rollout_* | head -1)
tar -czf box_height_v03_result.tar.gz \
  -C "$RUN" \
  box_height_summary.json samples.jsonl first_table_depth_mm.npz \
  sample_000_table_rgb.png sample_000_depth_preview.png \
  sample_000_depth_on_rgb.png
```

验证通过后，下一阶段才把 table depth 高度图用于真实抓取程序的保守避障，
并对所有估计障碍物高度额外增加至少 10 mm 安全余量。
